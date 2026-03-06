clear; clc;
load('D:\规范化数据集\cancersta.mat'); 

M = 5;              
NN = 5;             
L = 200;           
rho = 1e-4;         
max_iter = 30;     
use_quadprog = true;

scale_noise = 0.3;   
sigma_noise = 0.3;  

sigma_kernel = 1;   
n_candidates = [1,2];  
ratio_candidates = [0.4, 0.6, 0.8];  
C_candidates = [1e-1, 1, 1e1, 1e2, 1e3];
S_fixed = 4;        
u_fixed = 4;
v_fixed = -2;

rho_param = 1e-3;    
max_iter_param = 15;
val_folds = 2;       

m1 = size(mu, 1);
m2 = size(zi, 1);
p1 = floor(m1 / M);
p2 = floor(m2 / M);

AAA = zeros(NN, 1);
best_ratio = 0.6;
best_C_pos = 1;
best_C_neg = 1;

for jj = 1:NN
    acc = zeros(M, 1);
    for t = 1:M
         X1_train = [mu(1:p1*(t-1), :); mu(p1*t+1:end, :)];
        X2_train = [zi(1:p2*(t-1), :); zi(p2*t+1:end, :)];
        T = [mu(p1*t-p1+1:p1*t, :); zi(p2*t-p2+1:p2*t, :)];
        test_labels = [ones(p1, 1); -ones(p2, 1)];

        N1 = size(X1_train, 1);
        N2 = size(X2_train, 1);
        if N1 == 0 || N2 == 0
            error('某类训练样本为空');
        end

        dim1 = randperm(N1);
        num_noise1 = floor(N1 * scale_noise);
        if num_noise1 > 0
            X1_train(dim1(1:num_noise1), :) = X1_train(dim1(1:num_noise1), :) ...
                + normrnd(0, sigma_noise, num_noise1, size(X1_train,2));
        end
        dim2 = randperm(N2);
        num_noise2 = floor(N2 * scale_noise);
        if num_noise2 > 0
            X2_train(dim2(1:num_noise2), :) = X2_train(dim2(1:num_noise2), :) ...
                + normrnd(0, sigma_noise, num_noise2, size(X2_train,2));
        end

   
        X_all_train = [X1_train; X2_train];
        mu_all = mean(X_all_train);
        sigma_all = std(X_all_train);
        sigma_all(sigma_all == 0) = 1;
        X1_train = (X1_train - mu_all) ./ sigma_all;
        X2_train = (X2_train - mu_all) ./ sigma_all;
        T = (T - mu_all) ./ sigma_all;

   
        if t == 1 && jj == 1
            fprintf('正在进行参数选择（随机搜索）...\n');
            [ratio_grid, C_pos_grid, C_neg_grid] = ndgrid(ratio_candidates, C_candidates, C_candidates);
            all_combs = [ratio_grid(:), C_pos_grid(:), C_neg_grid(:)];
            total_combs = size(all_combs, 1);
            rand_idx = randperm(total_combs);
            num_samples = min(15, total_combs);
            sampled_combs = all_combs(rand_idx(1:num_samples), :);

            best_acc = 0;
            val_size1 = floor(N1 / val_folds);
            val_size2 = floor(N2 / val_folds);

            for s_idx = 1:num_samples
                ratio_try = sampled_combs(s_idx, 1);
                C_pos_try = sampled_combs(s_idx, 2);
                C_neg_try = sampled_combs(s_idx, 3);
                s_try = S_fixed * ratio_try;

                acc_val_sum = 0;
                for v = 1:val_folds
                    X1_val = X1_train((v-1)*val_size1+1:v*val_size1, :);
                    X2_val = X2_train((v-1)*val_size2+1:v*val_size2, :);
                    X1_tr = [X1_train(1:(v-1)*val_size1, :); X1_train(v*val_size1+1:end, :)];
                    X2_tr = [X2_train(1:(v-1)*val_size2, :); X2_train(v*val_size2+1:end, :)];

                    [beta1_val, beta2_val] = train_ssptelm_kernel(X1_tr, X2_tr, ...
                        C_pos_try, C_neg_try, C_pos_try, C_neg_try, ...
                        1, S_fixed, s_try, u_fixed, v_fixed, sigma_kernel, ...
                        rho_param, max_iter_param, use_quadprog);

                  
                    H_val = gaussian_kernel([X1_val; X2_val], [X1_tr; X2_tr], sigma_kernel);
                    dist1 = abs(H_val * beta1_val);
                    dist2 = abs(H_val * beta2_val);
                    pred_val = ones(size(H_val,1),1);
                    pred_val(dist2 < dist1) = -1;
                    true_labels = [ones(size(X1_val,1),1); -ones(size(X2_val,1),1)];
                    acc_val = sum(pred_val == true_labels) / length(true_labels);
                    acc_val_sum = acc_val_sum + acc_val;
                end
                acc_val_avg = acc_val_sum / val_folds;
                if acc_val_avg > best_acc
                    best_acc = acc_val_avg;
                    best_ratio = ratio_try;
                    best_C_pos = C_pos_try;
                    best_C_neg = C_neg_try;
                end
            end
            fprintf('参数选择完成：s/S=%.1f, C_pos=%.2e, C_neg=%.2e, 验证准确率=%.4f\n', ...
                best_ratio, best_C_pos, best_C_neg, best_acc);
        end

        n = 1;
        s = S_fixed * best_ratio;
        C1 = best_C_pos; C3 = best_C_pos;
        C2 = best_C_neg; C4 = best_C_neg;
        u = u_fixed; v = v_fixed;

        [beta1, beta2] = train_ssptelm_kernel(X1_train, X2_train, ...
            C1, C2, C3, C4, n, S_fixed, s, u, v, sigma_kernel, ...
            rho, max_iter, use_quadprog);

        % 测试
        H_test = gaussian_kernel(T, [X1_train; X2_train], sigma_kernel);
        dist1 = abs(H_test * beta1);
        dist2 = abs(H_test * beta2);
        pred = ones(size(T,1),1);
        pred(dist2 < dist1) = -1;

        acc(t) = sum(pred == test_labels) / length(test_labels);
    end
    AAA(jj) = mean(acc);
end

final_acc_mean = mean(AAA);
final_acc_std = std(AAA);
fprintf('\n最终结果: 平均准确率 = %.4f ± %.4f\n', final_acc_mean, final_acc_std);


function K = gaussian_kernel(X, Y, sigma)
    n1 = size(X,1);
    n2 = size(Y,1);
    XX = sum(X.*X, 2);
    YY = sum(Y.*Y, 2);
    XY = X * Y';
    K = exp(-(repmat(XX,1,n2) + repmat(YY',n1,1) - 2*XY) / (2*sigma^2));
end

function [beta1, beta2] = train_ssptelm_kernel(X1, X2, C1, C2, C3, C4, n, S, s, u, v, sigma, rho, max_iter, use_quadprog)
      N1 = size(X1, 1);
    N2 = size(X2, 1);
    N = N1 + N2;      
    
    X_all = [X1; X2];
    K_all = gaussian_kernel(X_all, X_all, sigma);
    
    % 提取子块
    K1 = K_all(1:N1, :);       % N1 x N
    K2 = K_all(N1+1:end, :);   % N2 x N
    K11 = K1(:, 1:N1);         % N1 x N1
    K12 = K1(:, N1+1:end);     % N1 x N2
    K21 = K2(:, 1:N1);         % N2 x N1
    K22 = K2(:, N1+1:end);     % N2 x N2
    
      K1T = K_all(:, 1:N1);      % N x N1
    K2T = K_all(:, N1+1:end);  % N x N2
    
    I_N1 = eye(N1);
    I_N2 = eye(N2);
    e1 = ones(N1, 1);
    e2 = ones(N2, 1);
    
       M11_blk = (1/C1) * K11 + (2*C1/C3) * I_N1;
    M12_blk = (1/C1) * K12;
    M13_blk = -M12_blk;
    M22_blk = (1/C1) * K22 + (1/(C3 * S^2)) * I_N2;
    M23_blk = -(1/C1) * K22 + (1/(C3 * S * s)) * I_N2;
    M33_blk = (1/C1) * K22 + (1/(C3 * s^2)) * I_N2;
    M1 = [M11_blk, M12_blk, M13_blk; M12_blk', M22_blk, M23_blk; M13_blk', M23_blk', M33_blk];
    M1 = M1 + 1e-8 * eye(size(M1));
    
    term1_coef = (1/C3) * [-K21, -K22, K22];  % N2 x (N1+2N2)
    lb1 = [-inf(N1,1); zeros(2*N2,1)];
    ub1 = [inf(N1,1); inf(2*N2,1)];
    

    M11_2 = (1/C2) * K22 + (C2/C4) * I_N2;
    M12_2 = -(1/C2) * K21;
    M13_2 = (1/C2) * K21;
    M22_2 = (1/C2) * K11 + (1/(C4 * S^2)) * I_N1;
    M23_2 = -(1/C2) * K11 + (1/(C4 * S * s)) * I_N1;
    M33_2 = (1/C2) * K11 + (1/(C4 * s^2)) * I_N1;
    M2 = [M11_2, M12_2, M13_2; M12_2', M22_2, M23_2; M13_2', M23_2', M33_2];
    M2 = M2 + 1e-8 * eye(size(M2));
    
    term2_coef = (1/C4) * [K12, K11, -K11]; 
    lb2 = [-inf(N2,1); zeros(2*N1,1)];
    ub2 = [inf(N2,1); inf(2*N1,1)];
    
    if use_quadprog
        options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
    end
    
    % 初始化系数向量
    beta1 = zeros(N, 1);
    beta2 = zeros(N, 1);
    
    for iter = 1:max_iter
                T1 = e2 + K2 * beta1;
        theta1 = zeros(N2, 1);
        if n == 1
            theta1(T1 >= u) = -C1 * S * T1(T1 >= u);
            theta1(T1 >= v & T1 < u) = 0;
            theta1(T1 < v) = -C1 * s * T1(T1 < v);
        else % n=2
            theta1(T1 >= u) = -C1 * S;
            theta1(T1 >= v & T1 < u) = 0;
            theta1(T1 < v) = -C1 * s;
        end
        f1 = term1_coef' * theta1;  % (N1+2N2) x 1
        
        if use_quadprog
            x1 = quadprog(M1, f1, [], [], [], [], lb1, ub1, [], options);
        else
            x1 = qpSOR(M1, f1, lb1, ub1);
        end
        if isempty(x1), break; end
        gamma1 = x1(1:N1);
        alpha1 = x1(N1+1:N1+N2);
        lambda1 = x1(N1+N2+1:end);
        
        beta1_new = (1/C3) * (K2T * theta1 - K1T * gamma1 - K2T * alpha1 + K2T * lambda1);
        
              T2 = e1 - K1 * beta2;
        theta2 = zeros(N1, 1);
        if n == 1
            theta2(T2 >= u) = -C2 * S * T2(T2 >= u);
            theta2(T2 >= v & T2 < u) = 0;
            theta2(T2 < v) = -C2 * s * T2(T2 < v);
        else
            theta2(T2 >= u) = -C2 * S;
            theta2(T2 >= v & T2 < u) = 0;
            theta2(T2 < v) = -C2 * s;
        end
        f2 = term2_coef' * theta2;  % (N2+2N1) x 1
        
        if use_quadprog
            x2 = quadprog(M2, f2, [], [], [], [], lb2, ub2, [], options);
        else
            x2 = qpSOR(M2, f2, lb2, ub2);
        end
        if isempty(x2), break; end
        gamma2 = x2(1:N2);
        alpha2 = x2(N2+1:N2+N1);
        lambda2 = x2(N2+N1+1:end);
        
        beta2_new = (1/C4) * (K1T * theta2 + K2T * gamma2 + K1T * alpha2 - K1T * lambda2);
        
        % 检查收敛
        if norm(beta1_new - beta1) < rho && norm(beta2_new - beta2) < rho
            beta1 = beta1_new;
            beta2 = beta2_new;
            break;
        else
            beta1 = beta1_new;
            beta2 = beta2_new;
        end
    end
end


function x = qpSOR(H, f, lb, ub, omega, tol, maxit)
    if nargin < 5, omega = 1.5; end
    if nargin < 6, tol = 1e-6; end
    if nargin < 7, maxit = 1000; end
    n = length(f);
    x = zeros(n, 1);
    x_old = x;
    for it = 1:maxit
        for i = 1:n
            s = H(i,i);
            if abs(s) < 1e-12, continue; end
            r = H(i,:) * x + f(i);
            x(i) = x(i) - omega * r / s;
            x(i) = max(lb(i), min(ub(i), x(i)));
        end
        if norm(x - x_old, inf) < tol
            break;
        end
        x_old = x;
    end
end