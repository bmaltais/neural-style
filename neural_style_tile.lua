require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-multigpu_strategy', '', 'Index of layers to split the network across GPUs')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-init_image', '')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)
cmd:option('-lbfgs_num_correction', 0)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

-- Multi-resolution options
cmd:option('-expand_small', false, "Internally expand small images to 'tile_size' (slower and maybe doesn't change much).")
cmd:option('-overlap_times', 2, 'Number of fragments to mix for seamless stitching. Processing is squared times longer.')
cmd:option('-scale_down', 1, 'Starting scaling factor (0.5, for example).')
cmd:option('-scale_up', 1, "Final scaling factor (3.0 isn't bad).")
cmd:option('-scale_steps', 1, 'Upscale steps, 1 for auto (scale <= 2).')
cmd:option('-tile_size', 384, 'Maximum height / width of the tile image')


local function main(params)
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local dtype, multigpu = setup_gpu(params)
  local loadcaffe_backend = params.backend
  if loadcaffe_backend == 'clnn' then loadcaffe_backend = 'nn' end


  -- Loading images
  local content_image = image.load(params.content_image, 3)
  print("\n\nOriginal content image size (wxh) ", content_image:size(3), "x", content_image:size(2))

  content_image = image.scale(content_image, params.image_size, 'bilinear')

  local Ch = content_image:size(2)
  local Cw = content_image:size(3)
  print("Resized  content image size (wxh) ", Cw, "x", Ch)

  local content_image_caffe = preprocess(content_image):float()

  -- local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local i = image.load(img_path, 3)
    print(" ")
    print("Original style image size (wxh) ", i:size(3), "x", i:size(2))

    local Sh = i:size(2)
    local Sw = i:size(3)

    local style_size = 0
    local resizeStyle = 1

    local Cr = Cw / Ch
    local Sr = Sw / Sh

    if (Cr >= Sr) and Cr >= 1 and Sr >= 1 then
      style_size = Cw
      if style_size > Sw then 
        style_size = Sw
        resizeStyle = 0
      end
    end

    if (Cr <= Sr) and Cr >= 1 and Sr >= 1 then
      style_size = Cw
      if style_size > Sw then 
        style_size = Sw
        resizeStyle = 0
      end
    end

    if (Cr >= Sr) and Cr >= 1 and Sr <= 1 then
      style_size = Cw / Sr
      if Cw >= Sw then 
        style_size = Sw / Sr
        resizeStyle = 0
      end
    end

    if (Cr <= Sr) and Cr <= 1 and Sr <= 1 then
      style_size = Cw / Sr
      if Cw >= Sw then 
        style_size = Sw / Sr
        resizeStyle = 0
      end
    end

    if (Cr >= Sr) and Cr <= 1 and Sr <= 1 then
      style_size = Cw / Sr
      if Cw >= Sw then 
        style_size = Sw / Sr
        resizeStyle = 0
      end
    end

    if (Cr <= Sr) and Cr <= 1 and Sr >= 1 then
      style_size = Cw
      if style_size > Sw then 
        style_size = Sw
        resizeStyle = 0
      end
    end

    -- Check if style image need to be resized. If not don't resize it for nothing
    if resizeStyle == 1 then
      i = image.scale(i, style_size, 'bilinear')
    else
      print("Style image will not need to be resized beyond it's current size")
    end

    local Sh = i:size(2)
    local Sw = i:size(3)
    print("Resized  style image size (wxh) ", Sw, "x", Sh)

    local img_caffe = preprocess(i):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Functions for image splitting / merging
  local function image_part(full_image, x, y, width)  -- Coordinates are 1...w/h, space beyond source image is stretched

    local expand_small = params.expand_small or false -- expand small images to tile_size
    local minimal_size = 160 -- some models corrupt smaller images

    local sw, sh = full_image:size(3), full_image:size(2)
    local pw, ph = math.min(width, sw - x + 1), math.min(width, sh - y + 1)
    local fw, fh
    if (sw > width) or expand_small then fw = width else fw = pw end
    if (sh > width) or expand_small then fh = width else fh = ph end
    fw, fh = math.max(fw, minimal_size), math.max(fh, minimal_size)
    local part = torch.Tensor(full_image:size(1), fh, fw):type(full_image:type())
    part[{{}, {1, ph}, {1, pw}}] = full_image[{{}, {y, y + ph - 1}, {x, x + pw - 1}}]
    if pw < fw then
      part[{{}, {1, ph}, {pw + 1, fw}}] = part[{{}, {1, ph}, {pw, pw}}]:expand(part:size(1), ph, fw - pw)
    end
    if ph < fh then
      part[{{}, {ph + 1, fh}, {1, fw}}] = part[{{}, {ph, ph}, {1, fw}}]:expand(part:size(1), fh - ph, fw)
    end
    return part:clone()
  end

  local function image_overlay(full_image, part, x, y, overlap)  -- Coordinates are 1...w/h
    local fragment_size = params.tile_size
    -- Making mask for full fragment
    local og = torch.linspace(0, 1, overlap + 2)[{{2, -2}}]
    local xb = og:view(1, -1):expand(fragment_size, overlap)
    local yb = xb:t()
    local mask = torch.ones(fragment_size, fragment_size)
    if overlap > 0 then
      if x > 1 then
        mask[{{}, {1, overlap}}] = xb
      end
      if y > 1 then
        if x > 1 then
          mask[{{1, overlap}, {}}] = torch.cmin(mask[{{1, overlap}, {}}], yb)
        else
          mask[{{1, overlap}, {}}] = yb
        end
      end
    end
    -- Cropping
    local iw, ih = full_image:size(3), full_image:size(2)
    local pw, ph = math.min(part:size(3), iw - x + 1), math.min(part:size(2), ih - y + 1)
    local front = part[{{}, {1, ph}, {1, pw}}]:double()
    local back = full_image[{{}, {y, y + ph - 1}, {x, x + pw - 1}}]:double()
    mask = mask[{{1, ph}, {1, pw}}]:contiguous():view(1, ph, pw):expand(front:size()) --:contiguous()
    -- Combining
    local comb = torch.cmul(front, mask):add(torch.cmul(back, 1 - mask)):type(full_image:type())
    full_image[{{}, {y, y + ph - 1}, {x, x + pw - 1}}] = comb
    return full_image:contiguous()
  end


  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
           '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")


  -- Loading network, inserting style and content loss modules
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)
  local next_content_idx, next_style_idx = 1, 1
  local net_clean = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):type(dtype)
    net_clean:add(tv_mod)
  end
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_dropout = (layer_type == 'nn.Dropout' or layer_type == 'nn.SpatialDropout')
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_dropout then
        local msg = 'Removing dropout at layer %d'
        print(string.format(msg, i))
      elseif is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):type(dtype)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net_clean:add(avg_pool_layer)
      else
        net_clean:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, norm):type(dtype)
        net_clean:add(loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, norm):type(dtype)
        net_clean:add(loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end
  if multigpu then
    net_clean = setup_multi_gpu(net_clean, params)
  end
--  net_clean:type(dtype)

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i = 1, #net_clean.modules do
    local module = net_clean.modules[i]
    local t = torch.type(module)
    if (t == 'nn.SpatialConvolutionMM' or t == 'nn.SpatialConvolution') then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  print(net_clean)


  local scaledown, scaleup, scalesteps, scalecurrent, scalestring
  local cur_iter = 0
  local img = nil
  local init_image = nil
  local content_image_caffe_scaled = content_image_caffe


  local function maybe_save(i, appendix)
    local should_save = params.save_iter > 0 and cur_iter % params.save_iter == 0
    should_save = should_save or cur_iter >= params.num_iterations
    if should_save then
      local disp = deprocess(i:double())
      disp = image.minmax{tensor = disp, min = 0, max = 1}
      local filename = build_filename(params.output_image, cur_iter)
      if cur_iter == params.num_iterations then
        filename = params.output_image
      end
      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end
      image.save(filename .. appendix, disp)
    end
  end


  local function process_img() -- img shoud be initialized

    local overlap_times = params.overlap_times -- Processing is squared times longer
    local sub_iter = math.ceil(params.num_iterations / 3)  --math.ceil(math.sqrt(params.num_iterations))  --30
    local overlap = math.floor(params.tile_size * (overlap_times - 1) / overlap_times)

    if params.optimizer == 'lbfgs' then
      print('Running optimization with L-BFGS')
    elseif params.optimizer == 'adam' then
      print('Running optimization with ADAM')
    else
      error(string.format('Unrecognized optimizer "%s"', params.optimizer))
    end

    for iter_group = 1, params.num_iterations, sub_iter do
      -- Processing fragments
      local img_clean = img:clone()
      local fragment_counter = 0
      local fragments_total = math.ceil(math.max(1, img_clean:size(2) - overlap) / (params.tile_size - overlap)) *
                              math.ceil(math.max(1, img_clean:size(3) - overlap) / (params.tile_size - overlap)) -- ceil((b-a+1)/step)
      for fy = 1, math.max(1, img_clean:size(2) - overlap), params.tile_size - overlap do
        local img_row = torch.DoubleTensor(img:size(1), params.tile_size, img:size(3))
        for fx = 1, math.max(1, img_clean:size(3) - overlap), params.tile_size - overlap do
          fragment_counter = fragment_counter + 1
          print("Processing image part #" .. fragment_counter .. " of " .. fragments_total .. " ([" .. fx .. ", " .. fy .. "] of " .. content_image_caffe_scaled:size(3) .. "x" .. content_image_caffe_scaled:size(2) .. ", scaling factor " .. scalestring .. ").")

          -- Replicating network
          print("Resetting network.")
          local net = net_clean:clone():type(dtype)
          local content_losses, style_losses = {}, {}
          for i = 1, #net do
            local layer = net:get(i)
            local layer_type = torch.type(layer)
            if (layer_type == 'nn.ContentLoss') then
              table.insert(content_losses, layer)
            end
            if (layer_type == 'nn.StyleLoss') then
              table.insert(style_losses, layer)
            end
          end

          -- Loading parts of content image, scaled to current resolution
          local content_fragment = image_part(content_image_caffe_scaled, fx, fy, params.tile_size)

          -- Capture content targets
          for i = 1, #style_losses do
            style_losses[i].mode = 'none'
          end
          for i = 1, #content_losses do
            content_losses[i].mode = 'capture'
          end
          print 'Capturing content targets'
          net:forward(content_fragment:type(dtype))
          -- Capture style targets
          for i = 1, #content_losses do
            content_losses[i].mode = 'none'
          end
          for i = 1, #style_images_caffe do
            print(string.format('Capturing style target %d', i))
            for j = 1, #style_losses do
              style_losses[j].mode = 'capture'
              style_losses[j].blend_weight = style_blend_weights[i]
            end
            net:forward(style_images_caffe[i]:type(dtype))
          end
          -- Set all loss modules to loss mode
          for i = 1, #content_losses do
            content_losses[i].mode = 'loss'
          end
          for i = 1, #style_losses do
            style_losses[i].mode = 'loss'
          end

          -- Loading parts of work image (processed in previous iterations)
          local img_fragment = image_part(img_clean, fx, fy, params.tile_size):type(dtype)

          -- Run it through the network once to get the proper size for the gradient
          -- All the gradients will come from the extra loss modules, so we just pass
          -- zeros into the top of the net on the backward pass.
          local y = net:forward(img_fragment)
          local dy = img_fragment.new(#y):zero()

          -- Declaring this here lets us access it in maybe_print
          local optim_state = nil

          local function maybe_print(loss)
            local verbose = (params.print_iter > 0 and cur_iter % params.print_iter == 0)
            if verbose then
              print(string.format('Iteration %d / %d', cur_iter, params.num_iterations))
              contentLoss = 0
              for i, loss_module in ipairs(content_losses) do
                print(string.format('  Content %d loss: %f', i, loss_module.loss))
                contentLoss = contentLoss + loss_module.loss
              end
              print(string.format('  Total content loss: %f', contentLoss))
              styleLoss = 0
              for i, loss_module in ipairs(style_losses) do
                print(string.format('  Style %d loss: %f', i, loss_module.loss))
                styleLoss = styleLoss + loss_module.loss
              end
              print(string.format('  Total style loss: %f', styleLoss))
              print(string.format('  Total loss: %f', loss))
            end
          end

          -- Function to evaluate loss and gradient. We run the net forward and
          -- backward to get the gradient, and sum up losses from the loss modules.
          -- optim.lbfgs internally handles iteration and calls this function many
          -- times, so we manually count the number of iterations to handle printing
          -- and saving intermediate results.
          local function feval(x)
            net:forward(x)
            local grad = net:updateGradInput(x, dy)
            local loss = 0
            for _, mod in ipairs(content_losses) do
              loss = loss + mod.loss
            end
            for _, mod in ipairs(style_losses) do
              loss = loss + mod.loss
            end
            maybe_print(loss)
            collectgarbage()
            -- optim.lbfgs expects a vector for gradients
            return loss, grad:view(grad:nElement())
          end

          -- Run optimization
          if params.optimizer == 'lbfgs' then
            optim_state = {
              maxIter = 1,
              verbose=false,
              tolX=-1,
              tolFun=-1,
            }
            if params.lbfgs_num_correction > 0 then
              optim_state.nCorrection = params.lbfgs_num_correction
            end
          elseif params.optimizer == 'adam' then
            optim_state = {
              learningRate = params.learning_rate,
            }
          else
            error(string.format('Unrecognized optimizer "%s"', params.optimizer))
          end

          -- Running group of subiterations
          for si = 0, sub_iter - 1 do
            cur_iter = iter_group + si
            if params.optimizer == 'lbfgs' then
              local x, losses = optim.lbfgs(feval, img_fragment, optim_state)
            elseif params.optimizer == 'adam' then
              local x, losses = optim.adam(feval, img_fragment, optim_state)
            end
            if cur_iter >= params.num_iterations then break end
          end

          -- Making full row of fragments
          --if params.save_iter == 1 then
          --  maybe_save(img_fragment, "_fragment["fx..","..fy.."]_processed.png")
          --end

          if (overlap > 0) and (fx > 1) then
            local fragment_size = params.tile_size
            local iw, ih = img:size(3), img:size(2)
            local mw, mh = math.min(overlap, iw - fx + 1), math.min(fragment_size, ih - fy + 1)
            local xmask = torch.linspace(0, 1, overlap + 2)[{{2, -2}}]:view(1, 1, -1):expand(img_row:size(1), mh, overlap)
            -- Cropping
            local front = img_fragment[{{}, {1, mh}, {1, mw}}]:double()
            local back = img_row[{{}, {1, mh}, {fx, fx + mw - 1}}]:double()
            xmask = xmask[{{}, {}, {1, mw}}] --:contiguous()
            -- Combining
            img_row[{{}, {1, mh}, {fx, fx + mw - 1}}] = torch.cmul(front, xmask):add(torch.cmul(back, 1 - xmask)):type(img_row:type())
            local cw = math.min(fragment_size, iw - fx + 1) - overlap
            if cw > 0 then
              img_row[{{}, {1, mh}, {fx + overlap, fx + overlap + cw - 1}}] = img_fragment[{{}, {1, mh}, {1 + overlap, overlap + cw}}]:type(img_row:type())
            end
          else
            local fragment_size = params.tile_size
            local iw, ih = img:size(3), img:size(2)
            local cw, ch = math.min(fragment_size, iw - fx + 1), math.min(fragment_size, ih - fy + 1)
            if cw > 0 then
              img_row[{{}, {1, ch}, {fx, fx + cw - 1}}] = img_fragment[{{}, {1, ch}, {1, cw}}]:type(img_row:type())
            end
          end

          net = nil
          content_losses, style_losses = nil, nil
          collectgarbage()
        end  -- fx

        -- Placing row to image
        if (overlap > 0) and (fy > 1) then
          local iw, ih = img:size(3), img:size(2)
          local mw, mh = img_row:size(3), math.min(overlap, ih - fy + 1)
          local ymask = torch.linspace(0, 1, overlap + 2)[{{2, -2}}]:view(1, -1, 1):expand(img_row:size(1), overlap, mw)
          -- Cropping
          local front = img_row[{{}, {1, mh}, {}}]:double()
          local back = img[{{}, {fy, fy + mh - 1}, {}}]:double()
          ymask = ymask[{{}, {1, mh}, {}}] --:contiguous()
          -- Combining
          img[{{}, {fy, fy + mh - 1}, {}}] = torch.cmul(front, ymask):add(torch.cmul(back, 1 - ymask)):type(img:type())
          local ch = math.min(img_row:size(2), ih - fy + 1) - overlap
          if ch > 0 then
            img[{{}, {fy + overlap, fy + overlap + ch - 1}, {}}] = img_row[{{}, {overlap + 1, overlap + ch}, {}}]:type(img:type())
          end
        else
          local fragment_size = params.tile_size
          local iw, ih = img:size(3), img:size(2)
          local cw, ch = img_row:size(3), math.min(img_row:size(2), ih - fy + 1)
          if ch > 0 then
            img[{{}, {fy, fy + ch - 1}, {}}] = img_row[{{}, {1, ch}, {}}]:type(img:type())
          end
        end

        img_row = nil
        if params.save_iter == 1 then
          maybe_save(img, "_s["..scalestring.."]-r["..fy.."].png")
        end
        collectgarbage()
      end  -- fy

      if params.save_iter == 1 then
        maybe_save(img, "_s["..scalestring.."].png")
      end
    end -- iteration group
  end -- process_img()


  scaledown  = params.scale_down  --0.5
  scaleup    = params.scale_up    --2.0
  scalesteps = params.scale_steps --3

  local scale_eps = 1e-10
  if scaledown < 0 then
    scaledown = -scaledown / math.max(content_image:size(2), content_image:size(3)) + scale_eps
  end
  if scaleup < 0 then
    scaleup = -scaleup / math.max(content_image:size(2), content_image:size(3)) + scale_eps
  end

  if (scaledown > scaleup) then
    scalecurrent = scaleup
    scaleup = scaledown
    scaledown = scalecurrent
  end
  if scalesteps < 2 then
    if (scaleup / scaledown) > 2 then
      scalesteps = math.ceil(math.log(scaleup / scaledown) / math.log(2)) + 1
    else
      scalesteps = 2
    end
  end

  if params.init_image ~= '' then
    init_image = image.load(params.init_image, 3)
    local H, W = content_image:size(2), content_image:size(3)
    init_image = image.scale(init_image, W, H, 'bilinear')
    init_image = preprocess(init_image):float()
  end

  -- Initializing the image
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    if init_image then
      img = init_image:clone()
    else
      img = content_image_caffe:clone()
    end
  else
    error('Invalid init type')
  end
  img = img:type(dtype)

  scalecurrent = (math.log(scaleup) - math.log(scaledown)) / (scalesteps - 1)
  if scalecurrent == 0 then
    scalecurrent = 1
    scalesteps = "simple scaling."
  else
    scalesteps = scalesteps .. "-step scaling."
  end
  print( "Scale: " .. scaledown .. "-" .. scaleup .. ", " .. scalesteps )
  --print( math.log(scaledown), math.log(scaleup) + scale_eps, scalecurrent )
  for scalelog = math.log(scaledown), math.log(scaleup) + scale_eps, scalecurrent do
    scalecurrent = math.min(math.exp(scalelog), scaleup)
    scalestring  = string.format('%0.03f', scalecurrent)
    local H, W = math.max(math.floor(content_image:size(2) * scalecurrent), 1), math.max(math.floor(content_image:size(3) * scalecurrent), 1)
    img = image.scale(img:double(), W, H, 'bilinear')
    if (scalecurrent > 1.0) then
      content_image_caffe_scaled = img:clone():float()
    elseif (scalecurrent < 1.0) then
      content_image_caffe_scaled = image.scale(content_image_caffe, W, H, 'bilinear')
    else -- no scaling
      content_image_caffe_scaled = content_image_caffe
    end
    img = img:type(dtype)
    process_img()
  end

  maybe_save(img, "")

  print("Done.")

end -- main()


function setup_gpu(params)
  local multigpu = false
  if params.gpu:find(',') then
    multigpu = true
    params.gpu = params.gpu:split(',')
    for i = 1, #params.gpu do
      params.gpu[i] = tonumber(params.gpu[i]) + 1
    end
  else
    params.gpu = tonumber(params.gpu) + 1
  end
  local dtype = 'torch.FloatTensor'
  if multigpu or params.gpu > 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      if multigpu then
        cutorch.setDevice(params.gpu[1])
      else
        cutorch.setDevice(params.gpu)
      end
      dtype = 'torch.CudaTensor'
    else
      require 'clnn'
      require 'cltorch'
      if multigpu then
        cltorch.setDevice(params.gpu[1])
      else
        cltorch.setDevice(params.gpu)
      end
      dtype = torch.Tensor():cl():type()
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  return dtype, multigpu
end


function setup_multi_gpu(net, params)
  local DEFAULT_STRATEGIES = {
    [2] = {3},
  }
  local gpu_splits = nil
  if params.multigpu_strategy == '' then
    -- Use a default strategy
    gpu_splits = DEFAULT_STRATEGIES[#params.gpu]
    -- Offset the default strategy by one if we are using TV
    if params.tv_weight > 0 then
      for i = 1, #gpu_splits do gpu_splits[i] = gpu_splits[i] + 1 end
    end
  else
    -- Use the user-specified multigpu strategy
    gpu_splits = params.multigpu_strategy:split(',')
    for i = 1, #gpu_splits do
      gpu_splits[i] = tonumber(gpu_splits[i])
    end
  end
  assert(gpu_splits ~= nil, 'Must specify -multigpu_strategy')
  local gpus = params.gpu

  local cur_chunk = nn.Sequential()
  local chunks = {}
  for i = 1, #net do
    cur_chunk:add(net:get(i))
    if i == gpu_splits[1] then
      table.remove(gpu_splits, 1)
      table.insert(chunks, cur_chunk)
      cur_chunk = nn.Sequential()
    end
  end
  table.insert(chunks, cur_chunk)
  assert(#chunks == #gpus)

  local new_net = nn.Sequential()
  for i = 1, #chunks do
    local out_device = nil
    if i == #chunks then
      out_device = gpus[1]
    end
    new_net:add(nn.GPU(chunks[i], gpus[i], out_device))
  end

  return new_net
end


function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.scale(content, generated_y:size(3), generated_y:size(2), 'bilinear')
  content_uv = image.rgb2yuv(content_uv)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mode = 'none'
end

function ContentLoss:updateOutput(input)
  if self.mode == 'loss' then
    self.loss = self.crit:forward(input, self.target) * self.strength
  elseif self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end


local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')

function Gram:__init()
  parent.__init(self)
end

function Gram:updateOutput(input)
  assert(input:dim() == 3)
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.output:resize(C, C)
  self.output:mm(x_flat, x_flat:t())
  return self.output
end

function Gram:updateGradInput(input, gradOutput)
  assert(input:dim() == 3 and input:size(1))
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.gradInput:resize(C, H * W):mm(gradOutput, x_flat)
  self.gradInput:addmm(gradOutput:t(), x_flat)
  self.gradInput = self.gradInput:view(C, H, W)
  return self.gradInput
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = torch.Tensor()
  self.mode = 'none'
  self.loss = 0

  self.gram = nn.GramMatrix()
  self.blend_weight = nil
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  if self.mode == 'capture' then
    if self.blend_weight == nil then
      self.target = self.G
    elseif self.target:nElement() == 0 then
      self.target = torch.mul(self.G, self.blend_weight)
    else
      self.target:add(self.blend_weight, self.G)
    end
  elseif self.mode == 'loss' then
    self.loss = self.strength * self.crit:forward(self.G, self.target)
  end
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    local dG = self.crit:backward(self.G, self.target)
    dG:div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local params = cmd:parse(arg)
main(params)