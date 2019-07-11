load cc_data.mat
x_train=reshape(age',[1,1,1,size(age,1)]); 
y_train=reshape(data,[size(data,1)*size(data,2),size(data,3)])'; 


%% Load trained model for prediction

load('CC_shape_net.mat')
net.Layers
figure,
cmap = jet(length(1:100));
for x=1:100
   if ~isempty(intersect(x,age)) % exclude the training data
       YPredicted = predict(net,reshape(x,[1,1,1,1]));
       yu1=reshape(YPredicted,[2 size(data,2)]);
       plot(yu1(1,:),yu1(2,:),'Color', cmap(x,:))
       hold on;
   end
end
hold off;axis off;
colormap(cmap);
set(gcf,'color','w');

    
     
%% R^{2} statistic 
raw_data=reshape(data,[size(data,1)*size(data,2),size(data,3)]);
YPredicted = double(predict(net,x_train));
s_total=0; s_residual=0;
for i=1:size(data,3)
    s_total=s_total+sum((raw_data(:,i)-mean(raw_data,2)).^2);
    s_residual=s_residual+sum((raw_data(:,i)-YPredicted(i,:)').^2);
end
    
R2=1-s_residual/s_total;
disp(['The R^2 value is ', num2str(R2)])