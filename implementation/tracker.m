function results = tracker(params)
%% Yiming LI 20190503
%% Initialization
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
% Init position
pos = seq.init_pos(:)';
% context position
pos_left    = floor(params.init_pos_left);
pos_right    = floor(params.init_pos_right);
pos_bottom    = floor(params.init_pos_bottom);
pos_top    = floor(params.init_pos_top);

pos_lefttop =  params.init_pos_lefttop ;
pos_righttop = params.init_pos_righttop ;
pos_leftbottom = params.init_pos_leftbottom ;
pos_rightbottom = params.init_pos_rightbottom;

pos_context = [pos_left; pos_right; pos_bottom; pos_top; pos_lefttop; pos_righttop; pos_leftbottom; pos_rightbottom];

keystep = params.keystep;

target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

small_filter_sz{1} = floor(base_target_sz/(feature_cell_sz(1,1)));
small_filter_sz{2} = floor(base_target_sz/(feature_cell_sz(2,1)));

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;    
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y); 
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
cf_f = cell(num_feature_blocks, 1);
h_current_key = cell(num_feature_blocks, 1);
xl_context = cell(1, 8);
xlw_context = cell(1, 8);
xlf_context = cell(1, 8);
model_xf = cell(1, 2);
u_f = cell(1, 2);
h0 = cell(1, 2); %tmp storing keyfilter
S_xx_context = cell(1, 8);

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    tic();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
                                    
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);

    %%                   
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum + scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
            
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);

            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);            
            scale_change_factor = scaleFactors(sind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
                        
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    %%%% keyframe context learning
    if ((mod(seq.frame,keystep * params.Period) == 0)||(seq.frame==1))
    for p = 1:8
    % extract features
    xl_context{p} = extract_features(im, pos_context(p,:), currentScaleFactor, features, global_fparams, feature_extract_info);
    % do windowing of features
    xlw_context{p} =  cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_context{p}, cos_window, 'uniformoutput', false);
    % compute the fourier series
    xlf_context{p} = cellfun(@fft2, xlw_context{p}, 'uniformoutput', false);
    end
    end

    % train the CF model for each feature
    for k = 1: numel(xlf)
        if (seq.frame == 1)
            h_current_key{k} = zeros(size(xlf{k}));
            model_xf{k} = xlf{k};
        else
            model_xf{k} = ((1 - params.learningrate) * model_xf{k}) + (params.learningrate  * xlf{k});
        end
        
        % filter update of every Layer model
        g_f = single(zeros(size(xlf{k})));
        h_f = g_f;
        l_f = g_f;
        mu    = 1;
        betha = 20;
        mumax = 10000;
        gama = params.gama;
        i = 1;
            
        T = prod(filter_sz_cell{k});
        S_xx0 = sum(conj(model_xf{k}) .* model_xf{k}, 3);
        %keyframe context learning
        if ((mod(seq.frame,keystep * params.Period) == 0)||(seq.frame==1))
        for p =1:8
           S_xx_context{p} =  sum(conj(xlf_context{1,p}{k,1}) .* xlf_context{1,p}{k,1}, 3);
        end
        u_f{k}= model_xf{k} + params.yta1* (xlf_context{1,1}{k,1} + xlf_context{1,2}{k,1})...
               + params.yta2 * (xlf_context{1,3}{k,1} + xlf_context{1,4}{k,1}) ...
               + params.yta3* (xlf_context{1,5}{k,1} + xlf_context{1,6}{k,1} + xlf_context{1,7}{k,1} + xlf_context{1,8}{k,1});  
        S_xx = S_xx0 + params.yta1*(S_xx_context{1} + S_xx_context{2})...
             + params.yta2 * (S_xx_context{3} + S_xx_context{4})...
             + params.yta3 * (S_xx_context{5} + S_xx_context{6} + S_xx_context{7} + S_xx_context{8});
        S_uu = sum(conj(u_f{k}) .* model_xf{k}, 3);
        else
        S_xx = S_xx0;
        u_f{k}= model_xf{k};
        S_uu = sum(conj(u_f{k}) .* model_xf{k}, 3);
        end  
        %   ADMM
        while (i <= params.admm_iter)
                %   solve for G- please refer to the paper for more details
                B = S_xx + (T * mu);
                S_lx = sum(conj(u_f{k}) .* l_f, 3);
                S_hx = sum(conj(u_f{k}) .* h_f, 3);
                g_f = (((1/(T*mu)) * bsxfun(@times, yf{k}, model_xf{k})) - ((1/mu) * l_f) + h_f) - ...
                    bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, u_f{k}, (S_uu .* yf{k}))) - ((1/mu) * bsxfun(@times, u_f{k}, S_lx)) + (bsxfun(@times, u_f{k}, S_hx))), B);

                %   solve for H
                h = (T/((mu*T)+ params.admm_iter+ 2*gama))* (ifft2((mu*g_f) + l_f) )+ 2*gama.*h_current_key{k}/T;
                h0{k} = h;
                [sx,sy,h] = get_subwindow_no_window(h, floor(filter_sz_cell{k}/2) , small_filter_sz{k});
                t = gpuArray(zeros(filter_sz_cell{k}(1), filter_sz_cell{k}(2), size(h,3)));
                t(sx,sy,:) = h;
                h_f = fft2(t);

                %   update L
                l_f = l_f + (mu * (g_f - h_f));

                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);
                i = i+1;     
         end
         cf_f{k} = g_f;
         %keyfilter update
         if (mod(seq.frame,keystep) == 0 || seq.frame == 1)
             h_current_key{k} = h0{k};
         end
    end  
            
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    % update the position of context patches
    pos_left = pos - [target_sz(1,2)/2,0];
    pos_right = pos + [target_sz(1,2)/2,0];
    pos_bottom = pos - [0, target_sz(1,1)/2];
    pos_top = pos + [0, target_sz(1,1)/2];

    pos_lefttop = pos - [target_sz(1,2)/2,-1*target_sz(1,1)/2];
    pos_righttop = pos + [target_sz(1,2)/2,target_sz(1,1)/2];
    pos_leftbottom = pos - [target_sz(1,2)/2, target_sz(1,1)/2];
    pos_rightbottom = pos + [-1*target_sz(1,2)/2,target_sz(1,1)/2];
    
    pos_context = [pos_left; pos_right; pos_bottom; pos_top; pos_lefttop; pos_righttop; pos_leftbottom; pos_rightbottom];

    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %% Visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end      
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])                 
        drawnow
    end
end
[~, results] = get_sequence_results(seq);
disp(['fps: ' num2str(results.fps)])
