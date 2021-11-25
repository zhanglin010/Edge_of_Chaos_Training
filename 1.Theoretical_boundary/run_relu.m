clear
clc
close all

lb = [-inf, 0];
x0 = [0.1,0.1];
num_bins = 100;
J0s = zeros(1, 1.2*num_bins+1);
betas = zeros(1, num_bins);
integrals = zeros(num_bins, 1.2*num_bins+1);
integrals2 = zeros(num_bins, 1.2*num_bins+1);
options = optimoptions('lsqnonlin','Algorithm', 'trust-region-reflective','Display','final','FunctionTolerance',1e-10,'MaxFunctionEvaluations',500);
global J0 beta;

for i = 1:num_bins
    beta = i*2.0/num_bins;
    betas(1, i) = beta;
    for j = -num_bins:num_bins/5
        J0 = j*5.0/num_bins;
        J0s(1, j+num_bins+1) = J0;
        J = [beta, J0];
        disp(J);
        [x,resnorm,residual,exitflag,output] = lsqnonlin(@relu_integral,x0,lb,[],options);
        % disp(x);
        % disp(residual);
        integrand3 = @(z) exp(-z.^2/2).*1/sqrt(2.*pi);
        integ = integral(integrand3, -J0*x(1)/sqrt(x(2)), 999);
        integrals(i, j+num_bins+1) = ((abs((1/beta^2).*integ-1)/(abs((1/beta^2).*integ)+abs(1)))<0.02)*1;
        integrals2(i, j+num_bins+1) = abs((1/beta^2).*integ-1)/(abs((1/beta^2).*integ)+abs(1));
        % if abs(x(1))<20*abs(residual(1)) || abs(x(2))<20*abs(residual(2))
        % if abs(x(2))<20*abs(residual(2))
        % if exitflag == 0
        % integrals(i, j+num_bins+1) = 0.5;      
        % end
    end
end

subplot(2,1,1);
hmo = heatmap(J0s, flip(betas), flip(integrals));
title('\fontsize{22}Order-Chaos boundary');
hmo.XLabel = '\fontsize{16}J0/J';
hmo.YLabel = '\fontsize{16}1/J';

subplot(2,1,2);
heatmap(J0s, flip(betas), flip(integrals2));
title('\fontsize{16}Deviation of lhs of Eq.(3) from 1')

save('data_100bins_relu.mat', 'integrals')

function F = relu_integral(x)
%
global J0 beta
u = x(1);
q = x(2);
integrand1 = @(z) max(((J0/beta).*u+(1/beta).*sqrt(q).*z),0).*exp(-z.^2/2).*1/sqrt(2.*pi);
integrand2 = @(z) max(((J0/beta).*u+(1/beta).*sqrt(q).*z),0).^2.*exp(-z.^2/2).*1/sqrt(2.*pi);
F(1) = integral(integrand1, -inf, inf) - u;
F(2) = integral(integrand2, -inf, inf) - q;
end