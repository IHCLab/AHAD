function [HSI_sub, GT_sub] = crop(HSI,GT,dx,dy,shifting_pixel)
[H,W,~] = size(HSI);
central_start = shifting_pixel+1;
central_x_end = H-shifting_pixel;
central_y_end = W-shifting_pixel;

HSI_sub = HSI(central_start+dx:central_x_end+dx,central_start+dy:central_y_end+dy,:);
GT_sub  = GT(central_start+dx:central_x_end+dx,central_start+dy:central_y_end+dy);

end
