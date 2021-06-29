clear;
clc;

kernels_kai = getfield(load('/home/oa/code/python/VDNet-TPAMI/test_data/kernels_SISR/kernels_12.mat'), 'kernels');

kernels = zeros(15, 15, 8);

for ii = 1:8
    temp = kernels_kai{ii};
    temp = temp(6:20, 6:20);
    temp = temp / sum(temp(:));
    kernels(:, :, ii) = temp;
end

save('/home/oa/code/python/VDNet-TPAMI/test_data/kernels_SISR/kernels_8.mat', 'kernels');

