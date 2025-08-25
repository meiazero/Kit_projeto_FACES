% MLP 1-hidden layer classifier

function [STATS, TX_OK, W1, W2, R2_train_mean, R2_test_mean] = mlp1h(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H = 20;
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  epsn = 1e-8;

  % defaults
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  if ~isfield(config,'hidden_act'), config.hidden_act = 'sigmoid'; end
  if ~isfield(config,'opt_variant'), config.opt_variant = 'gd'; end
  if ~isfield(config,'eta'), config.eta = 0.01; end
  if ~isfield(config,'epochs'), config.epochs = 200; end
  if ~isfield(config,'mu'), config.mu = 0.9; end
  if ~isfield(config,'rho'), config.rho = 0.9; end
  if ~isfield(config,'eps_opt'), config.eps_opt = 1e-8; end
  if ~isfield(config,'leaky_alpha'), config.leaky_alpha = 0.01; end

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % normalizacao
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

    % one-hot
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain_oh(i, Train(i,end)) = 1; end

    % inicializacao
    if any(strcmp(config.hidden_act, {'relu','leakyrelu'}))
      W1 = randn(p, H) * sqrt(2/p);
    else
      W1 = randn(p, H) * sqrt(1/p);
    end
    b1 = zeros(1, H);
    W2 = randn(H, K) * sqrt(1/H);
    b2 = zeros(1, K);

    % estados
    V1 = zeros(size(W1)); V2 = zeros(size(W2));
    S1 = zeros(size(W1)); S2 = zeros(size(W2));

    % treino
    for e=1:config.epochs
      Z1 = Xtrain * W1 + b1;
      A1 = forward(Z1, config.hidden_act, config.leaky_alpha);

      Z2 = A1 * W2 + b2;
      A2 = softmax(Z2);

      dZ2 = (A2 - Ytrain_oh) / Ntrain;
      dW2 = A1' * dZ2; db2 = sum(dZ2,1);

      dZ1 = (dZ2 * W2') .* backward(Z1, A1, config.hidden_act, config.leaky_alpha);
      dW1 = Xtrain' * dZ1; db1 = sum(dZ1,1);

      switch config.opt_variant
        case 'gd'
          W1 = W1 - config.eta * dW1; b1 = b1 - config.eta * db1;
          W2 = W2 - config.eta * dW2; b2 = b2 - config.eta * db2;
        case 'momentum'
          V1 = config.mu * V1 + config.eta * dW1; V2 = config.mu * V2 + config.eta * dW2;
          W1 = W1 - V1; W2 = W2 - V2;
          b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2;
        case 'nesterov'
          W1_look = W1 - config.mu * V1; W2_look = W2 - config.mu * V2;
          % forward lookahead (aprox)
          Z1_la = Xtrain * W1_look + b1;
          A1_la = forward(Z1_la, config.hidden_act, config.leaky_alpha);
          Z2_la = A1_la * W2_look + b2;
          A2_la = softmax(Z2_la);

          dZ2_la = (A2_la - Ytrain_oh) / Ntrain;
          dW2_la = A1_la' * dZ2_la; db2_la = sum(dZ2_la,1);
          dZ1_la = (dZ2_la * W2_look') .* backward(Z1_la, A1_la, config.hidden_act, config.leaky_alpha);
          dW1_la = Xtrain' * dZ1_la; db1_la = sum(dZ1_la,1);

          V1 = config.mu * V1 + config.eta * dW1_la; V2 = config.mu * V2 + config.eta * dW2_la;
          W1 = W1 - V1; W2 = W2 - V2;
          b1 = b1 - config.eta * db1_la; b2 = b2 - config.eta * db2_la;
        case 'rmsprop'
          S1 = config.rho * S1 + (1-config.rho) * (dW1.^2);
          S2 = config.rho * S2 + (1-config.rho) * (dW2.^2);
          W1 = W1 - (config.eta ./ sqrt(S1 + config.eps_opt)) .* dW1;
          W2 = W2 - (config.eta ./ sqrt(S2 + config.eps_opt)) .* dW2;
          b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2;
        otherwise
          W1 = W1 - config.eta * dW1; b1 = b1 - config.eta * db1;
          W2 = W2 - config.eta * dW2; b2 = b2 - config.eta * db2;
      end
    end % epochs

    % --- Compute R2 on training data
    y_true_train = Train(:,end);
    Z1_tr = Xtrain * W1 + b1;
    A1_tr = forward(Z1_tr, config.hidden_act, config.leaky_alpha);
    Z2_tr = A1_tr * W2 + b2;
    A2_tr = softmax(Z2_tr);
    [~, pred_train] = max(A2_tr, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - (SSres_train / SStot_train);
    % --- Test evaluation
    Z1t = Xtest * W1 + b1;
    switch config.hidden_act
      case 'sigmoid', A1t = 1./(1+exp(-Z1t));
      case 'tanh',    A1t = tanh(Z1t);
      case 'relu',    A1t = max(Z1t,0);
      case 'leakyrelu', A1t = max(Z1t,0) + config.leaky_alpha * min(Z1t,0);
      otherwise, A1t = 1./(1+exp(-Z1t));
    end
    Z2t = A1t * W2 + b2;
    A2t = softmax(Z2t);
    [~, pred] = max(A2t, [], 2);

    TX_OK(r) = sum(pred == Test(:,end)) / size(Test,1) * 100;

    % Coeficiente de determinação (R^2) entre rótulos e predições (test)
    y_true_test = Test(:,end);
    y_pred_test = pred;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - (SSres_test / SStot_test);
  end % repeats

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);

  fprintf('mlp1h: normalization: %s, hidden_act: %s, opt_variant: %s, eta: %.3f, epochs: %d\n', config.normalization, config.hidden_act, config.opt_variant, config.eta, config.epochs);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction


function A = forward(Z, act, alpha)
  switch act
    case 'sigmoid',    A = 1./(1+exp(-Z));
    case 'tanh',       A = tanh(Z);
    case 'relu',       A = max(Z,0);
    case 'leakyrelu',  A = max(Z,0) + alpha*min(Z,0);
    otherwise,         A = 1./(1+exp(-Z));
  end
endfunction

function D = backward(Z, A, act, alpha)
  switch act
    case 'sigmoid',    D = A .* (1 - A);
    case 'tanh',       D = 1 - A.^2;
    case 'relu',       D = (Z > 0);
    case 'leakyrelu',  D = (Z > 0) + alpha*(Z <= 0);
    otherwise,         D = A .* (1 - A);
  end
endfunction

function S = softmax(Z)
  Zs = Z - max(Z,[],2);
  EZ = exp(Zs);
  S = EZ ./ sum(EZ,2);
endfunction
