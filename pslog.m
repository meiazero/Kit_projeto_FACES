% Perceptron Logistic Classifier
function [STATS, TX_OK, W, R2_train_mean, R2_test_mean] = pslog(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  epsn = 1e-8;

  % defaults
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  if ~isfield(config,'opt_variant'), config.opt_variant = 'gd'; end
  if ~isfield(config,'eta'), config.eta = 0.01; end
  if ~isfield(config,'epochs'), config.epochs = 200; end
  if ~isfield(config,'mu'), config.mu = 0.9; end
  if ~isfield(config,'rho'), config.rho = 0.9; end
  if ~isfield(config,'eps_opt'), config.eps_opt = 1e-8; end

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % Normalizacao
    switch config.normalization
      case 'zscore'
        m = mean(Xtrain_raw,1); s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s; Xtest = (Xtest_raw - m) ./ s;
      case 'minmax01'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng; Xtest = (Xtest_raw - mn) ./ rng;
      case 'minmax11'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn)./rng)-1; Xtest = 2*((Xtest_raw - mn)./rng)-1;
      case 'none'
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
      otherwise
        Xtrain = (Xtrain_raw - mean(Xtrain_raw,1)) ./ max(std(Xtrain_raw,0,1), epsn);
        Xtest  = (Xtest_raw - mean(Xtrain_raw,1)) ./ max(std(Xtrain_raw,0,1), epsn);
    end

    % One-hot
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain_oh(i, Train(i,end)) = 1; end

    Xb = [ones(Ntrain,1) Xtrain];
    % Inicializacao (Glorot-like)
    W = randn(p+1, K) * sqrt(2/(p+K));

    % estados otimizadores
    V = zeros(size(W));
    S = zeros(size(W));

    % treino
    for e=1:config.epochs
      Z = Xb * W;
      P = softmax_rows(Z);
      G = (Xb' * (P - Ytrain_oh)) / Ntrain;

      switch config.opt_variant
        case 'gd'
          W = W - config.eta * G;
        case 'momentum'
          V = config.mu * V + config.eta * G;
          W = W - V;
        case 'nesterov'
          W_look = W - config.mu * V;
          P_la = softmax_rows(Xb * W_look);
          G_la = (Xb' * (P_la - Ytrain_oh)) / Ntrain;
          V = config.mu * V + config.eta * G_la;
          W = W - V;
        case 'rmsprop'
          S = config.rho * S + (1 - config.rho) * (G.^2);
          W = W - (config.eta ./ sqrt(S + config.eps_opt)) .* G;
        otherwise
          W = W - config.eta * G;
      end
    end

    % --- Compute R2 on training data
    y_true_train = Train(:,end);
    Xb_train = Xb;
    Ztrain = Xb_train * W;
    [~, pred_train] = max(Ztrain, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - SSres_train / SStot_train;
    % --- Test evaluation
    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ztest = Xtest_b * W;
    [~, pred] = max(Ztest, [], 2);

    TX_OK(r) = sum(pred == Test(:,end)) / size(Test,1) * 100;

    % Coeficiente de determinação (R^2) entre rótulos e predições (test)
    y_true_test = Test(:,end);
    y_pred_test = pred;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - (SSres_test / SStot_test);
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);

  fprintf('pslog: normalization: %s, opt_variant: %s, eta: %.3f, epochs: %d\n', config.normalization, config.opt_variant, config.eta, config.epochs);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction

function S = softmax_rows(Z)
  Zs = Z - max(Z,[],2);
  EZ = exp(Zs);
  S = EZ ./ sum(EZ,2);
endfunction
