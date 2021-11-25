clear
clc
close all

lb = [-inf, 0];
x0 = [1, 1];
num_bins = 100;
J0s = zeros(1, fix(3*num_bins/2)+1);
betas = zeros(1, num_bins);
integrals = zeros(num_bins, fix(3*num_bins/2)+1);
integrals2 = zeros(num_bins, fix(3*num_bins/2)+1);
u_values = zeros(num_bins, fix(3*num_bins/2)+1);
q_values = zeros(num_bins, fix(3*num_bins/2)+1);
imagin_part = zeros(num_bins, fix(3*num_bins/2)+1);
options = optimoptions('lsqnonlin','Algorithm', 'trust-region-reflective','Display','final','FunctionTolerance',1e-10,'MaxFunctionEvaluations',500);
global J0 beta;

for i = 1:num_bins
    beta = i*2.0/num_bins;
    betas(1, i) = beta;
    for j = -num_bins/2:num_bins
        J0 = j*2.0/num_bins;
        J0s(1, j+num_bins/2+1) = J0;
        J = [beta, J0];
        disp(J);
        [x, resnorm,residual,exitflag,output] = lsqnonlin(@tanh_integral,x0,lb,[],options);
        integrand3 = @(z) sech((J0/beta).*x(1)+(1/beta).*sqrt(x(2)).*z).^4.*exp(-z.^2/2).*1/sqrt(2.*pi);
        integ = integral(integrand3, -inf, inf);
        integrals(i, j+num_bins/2+1) =((abs((1/beta^2).*integ-1)/(abs((1/beta^2).*integ)+abs(1)))<0.019)*1;
        integrals2(i, j+num_bins/2+1) = abs((1/beta^2).*integ-1)/(abs((1/beta^2).*integ)+abs(1));
        u_values(i, j+num_bins/2+1) = x(1);
        q_values(i, j+num_bins/2+1) = x(2);
    end
end

subplot(2,2,1);
heatmap(J0s, flip(betas), flip(integrals));
title('Order-Chaos boundary')

subplot(2,2,2);
heatmap(J0s, flip(betas), flip(integrals2));
title('\fontsize{16}Deviation of lhs of Eq.3 from 1')

subplot(2,2,3);
heatmap(J0s, flip(betas), flip(u_values));
title('Values of u')

subplot(2,2,4);
heatmap(J0s, flip(betas), flip(q_values));
title('Values of q')

save('data_100bins_tanh.mat', 'integrals')

function F = tanh_integral(x)
%
global J0 beta
u = x(1);
q = x(2);
integrand1 = @(z) tanh((J0/beta).*u+(1/beta).*sqrt(q).*z).*exp(-z.^2/2).*(1/sqrt(2.*pi));
integrand2 = @(z) tanh((J0/beta).*u+(1/beta).*sqrt(q).*z).^2.*exp(-z.^2/2).*(1/sqrt(2.*pi));
F(1) = integral(integrand1, -inf, inf) - u;
F(2) = integral(integrand2, -inf, inf) - q;
end