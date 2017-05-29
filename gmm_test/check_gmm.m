% Get GMM information
% disp('Given 1:');
[bm1, bs1, bw1, bm2, bs2, bw2] = get_msw('bg_gmm');
% disp('Given 2:');
[fm1, fs1, fw1, fm2, fs2, fw2] = get_msw('fg_gmm');

% Get points
bg_points = importdata('bg_points');
fg_points = importdata('fg_points');

% Set initial GMM information for fitting
bs = struct('mu', bm1, 'Sigma', bs1, 'ComponentProportion', bw1);
fs = struct('mu', fm1, 'Sigma', fs1, 'ComponentProportion', fw1);

% Fit GMM with points and old model
bg_GMModel = fitgmdist(bg_points, size(bm1,1), 'Start', bs);
% disp('Calculated');
% disp(['mu=', mat2str(bg_GMModel.mu), ', s=', mat2str(reshape(bg_GMModel.Sigma, [1,size(bm1,1)])), ...
%     ', w=', mat2str(bg_GMModel.ComponentProportion)]);
fg_GMModel = fitgmdist(fg_points, size(fm1,1), 'Start', fs);
% disp(['mu=', mat2str(fg_GMModel.mu), ', s=', mat2str(reshape(fg_GMModel.Sigma, [1,size(fm1,1)])), ...
%     ', w=', mat2str(fg_GMModel.ComponentProportion)]);
disp(['Calculated New: ', num2str(get_energy(fg_GMModel, bg_GMModel, fg_points, bg_points))])



% Check the given GMM models
old_bg = gmdistribution(bm1, bs1, bw1);
old_fg = gmdistribution(fm1, fs1, fw1);
disp(['Given Old: ', num2str(get_energy(old_fg, old_bg, fg_points, bg_points))])
new_bg = gmdistribution(bm2, bs2, bw2);
new_fg = gmdistribution(fm2, fs2, fw2);
disp(['Given New: ', num2str(get_energy(new_fg, new_bg, fg_points, bg_points))])


% Plot GMMs
% data_range=[0,300];
% X=(data_range(1):1:data_range(2)).';
% Y=pdf(old_bg,X);
% figure(2),plot(X,Y);
% hold on
% Y2=pdf(new_bg,X);
% plot(X,Y2);
% Y3=pdf(bg_GMModel,X);
% plot(X,Y3);
% hold off
% legend('given1', 'given2', 'calculated');