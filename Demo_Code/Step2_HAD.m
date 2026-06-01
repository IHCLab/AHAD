clear all;  clc; close all;

%% Settings
addpath(genpath('datasets'));
addpath(genpath('SuperRPCA_Function'));
lambda_a = 1; lambda_b = 0.1; threshold = 0.03; normalize = 0.3;
N = 2; superpixel_size = 12; dx=0; dy=0;

%% Load Data and Anti-Detection Signal
data_name ='abu-urban-4';
load(strcat('./datasets/',data_name,'.mat'))
load(strcat('./AHAD_Data/',data_name,'.mat'))
[Y_H, map] = crop(data_cube,map,dx,dy,Shifting_pixel_num);
Y_A = double(Y_H + Perturbation);

%% SuperRPCA Detection
SuperRPCA_R = SuperRPCA_detection(Y_H,lambda_a,lambda_b,threshold,normalize,N,superpixel_size);
SuperRPCA_A = SuperRPCA_detection(Y_A,lambda_a,lambda_b,threshold,normalize,N,superpixel_size);

AUC_up = roc(SuperRPCA_R,map,0);fprintf('AUC (Unperturbed): %f\n',round(AUC_up,4))
AUC_ap = roc(SuperRPCA_A,map,0);fprintf('AUC (AHAD-perturbed): %f\n',round(AUC_ap,4))

%% Visualization

fig=figure('name','SuperRPCA');
subplot(1,2,1)
imshow(ImGray2Pseudocolor(SuperRPCA_R, 'hot', 255));title('SuperRPCA (Unperturbed)');xlabel('ArmCBA:');
subplot(1,2,2)
imshow(ImGray2Pseudocolor(SuperRPCA_A, 'hot', 255));title('SuperRPCA (AHAD-perturbed)');xlabel((1-AUC_ap/AUC_up)*100);
          
%% Subprogram 1

function result = SuperRPCA_detection(data,lambda_a,lambda_b,threshold,normalize,N,superpixel_size)

 [B_CSRD, ~]  = CSRD_optimization(data, superpixel_size, normalize);
    
 [result, ~] = SuperRPCA_optimization(data, B_CSRD, lambda_a, lambda_b, N, threshold);
end
