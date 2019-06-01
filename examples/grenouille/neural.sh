th ../neural_style.lua -seed 100 \
-backend cudnn -cudnn_autotune \
-style_scale 1 -init image -normalize_gradients \
-image_size 384 -num_iterations 2500 -save_iter 50 \
-content_weight 100 -style_weight 1000 \
-style_image style.jpg \
-content_image content.jpg \
-output_image result-384.png \
-model_file ../../models/VGG_ILSVRC_19_layers.caffemodel -proto_file ../../models/VGG_ILSVRC_19_layers_deploy.prototxt \
-content_layers relu1_1,relu2_1,relu3_1,relu4_1,relu4_2,relu5_1 \
-style_layers relu3_1,relu4_1,relu4_2,relu5_1 \
-tv_weight 0.000085 -original_colors 0 && rm *_*0.png

th ../neural_style.lua -seed 100 \
-backend cudnn -cudnn_autotune \
-style_scale 1 -init image -normalize_gradients \
-image_size 512 -num_iterations 500 -save_iter 50 \
-content_weight 100 -style_weight 1000 \
-style_image style.jpg \
-content_image content.jpg \
-init_image result-384.png \
-output_image result-512.png \
-model_file ../../models/VGG_ILSVRC_19_layers.caffemodel -proto_file ../../models/VGG_ILSVRC_19_layers_deploy.prototxt \
-content_layers relu1_1,relu2_1,relu3_1,relu4_1,relu4_2,relu5_1 \
-style_layers relu3_1,relu4_1,relu4_2,relu5_1 \
-tv_weight 0.000085 -original_colors 0 && rm *_*0.png

th ../neural_style_tile.lua -seed 100 \
-backend cudnn -cudnn_autotune \
-style_scale 1 -init image -normalize_gradients \
-image_size 1200 -tile_size 500 -num_iterations 500 -save_iter 1 \
-content_weight 100 -style_weight 1000 \
-style_image style.jpg \
-content_image content.jpg \
-init_image result-512.png \
-output_image result-1200.png \
-model_file ../../models/VGG_ILSVRC_19_layers.caffemodel -proto_file ../../models/VGG_ILSVRC_19_layers_deploy.prototxt \
-content_layers relu1_1,relu2_1,relu3_1,relu4_1,relu4_2,relu5_1 \
-style_layers relu3_1,relu4_1,relu4_2,relu5_1 \
-tv_weight 0.000085 -original_colors 0 && rm *].png