% MLP 2-hidden layer classifier (com variações de normalização, ativações,
% otimizadores e estratégias de gradiente implementadas)

function [STATS, TX_OK, W1, W2, W3, R2_train_mean, R2_test_mean, rec_mean, prec_mean, f1_mean] = mlp2h(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H1 = 16; H2 = 8;
  TX_OK    = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test  = zeros(Nr,1);
  rec  = zeros(Nr,1); prec = zeros(Nr,1); f1 = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  % hidden activation variants (sigmoid, tanh, relu, leakyrelu)
  if ~isfield(config,'act1'),          config.act1 = 'sigmoid'; end
  if ~isfield(config,'act2'),          config.act2 = 'sigmoid'; end
  % optization variants (gd, adam, adagrad, rmsprop)
  if ~isfield(config,'opt_variant'),   config.opt_variant = 'gd'; end
  % gradient descent strategy (sgd, batch-gd, mini-batch-gd)
  if ~isfield(config,'gbds'),          config.gbds = 'batch-gd'; end

  if ~isfield(config,'batch_size'),    config.batch_size = 32; end
  if ~isfield(config,'eta'),           config.eta = 0.003; end
  if ~isfield(config,'epochs'),        config.epochs = 100; end
  if ~isfield(config,'mu'),            config.mu = 0.9; end
  if ~isfield(config,'rho'),           config.rho = 0.9; end       % rmsprop
  if ~isfield(config,'eps_opt'),       config.eps_opt = 1e-8; end
  if ~isfield(config,'leaky_alpha'),   config.leaky_alpha = 0.01; end
  if ~isfield(config,'beta1'),         config.beta1 = 0.9; end     % adam
  if ~isfield(config,'beta2'),         config.beta2 = 0.999; end   % adam

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % normalização
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
        m = mean(Xtrain_raw,1); s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s; Xtest = (Xtest_raw - m) ./ s;
    end

    % one-hot
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain_oh(i, Train(i,end)) = 1; end

    % inicialização (He ou Xavier simples)
    if any(strcmp(config.act1, {'relu','leakyrelu'})), W1 = randn(p, H1) * sqrt(2/p);
    else,                                               W1 = randn(p, H1) * sqrt(1/p);
    end
    b1 = zeros(1, H1);

    if any(strcmp(config.act2, {'relu','leakyrelu'})), W2 = randn(H1, H2) * sqrt(2/H1);
    else,                                               W2 = randn(H1, H2) * sqrt(1/H1);
    end
    b2 = zeros(1, H2);

    W3 = randn(H2, K) * sqrt(1/H2);
    b3 = zeros(1, K);

    % acumuladores para otimização
    V1 = zeros(size(W1)); V2 = zeros(size(W2)); V3 = zeros(size(W3));      % momentum / nesterov
    S1 = zeros(size(W1)); S2 = zeros(size(W2)); S3 = zeros(size(W3));      % rmsprop
    G1 = zeros(size(W1)); G2 = zeros(size(W2)); G3 = zeros(size(W3));      % adagrad
    mW1 = zeros(size(W1)); mW2 = zeros(size(W2)); mW3 = zeros(size(W3));   % adam (moment)
    vW1 = zeros(size(W1)); vW2 = zeros(size(W2)); vW3 = zeros(size(W3));   % adam (var)
    t = 0; % contador para Adam

    % treino
    for e=1:config.epochs
      % definição dos índices conforme estratégia de GD
      switch config.gbds
        case 'sgd'
          batch_size = 1;
        case 'mini-batch-gd'
          batch_size = min(config.batch_size, Ntrain);
        otherwise % 'batch-gd'
          batch_size = Ntrain;
      end
      perm = randperm(Ntrain);
      for bstart = 1:batch_size:Ntrain
        bend = min(bstart+batch_size-1, Ntrain);
        ids = perm(bstart:bend);
        Xb = Xtrain(ids,:);
        Yb = Ytrain_oh(ids,:);
        B = size(Xb,1);

        % forward
        Z1 = Xb * W1 + b1;
        A1 = forward(Z1, config.act1, config.leaky_alpha);

        Z2 = A1 * W2 + b2;
        A2 = forward(Z2, config.act2, config.leaky_alpha);

        Z3 = A2 * W3 + b3;
        A3 = softmax(Z3);

        % backward
        dZ3 = (A3 - Yb) / B;
        dW3 = A2' * dZ3; db3 = sum(dZ3,1);

        dA2 = dZ3 * W3';
        dZ2 = dA2 .* backward(Z2, A2, config.act2, config.leaky_alpha);
        dW2 = A1' * dZ2; db2 = sum(dZ2,1);

        dA1 = dZ2 * W2';
        dZ1 = dA1 .* backward(Z1, A1, config.act1, config.leaky_alpha);
        dW1 = Xb' * dZ1; db1 = sum(dZ1,1);

        % atualização conforme otimizador
        switch config.opt_variant
          case 'gd'
            W1 = W1 - config.eta * dW1; b1 = b1 - config.eta * db1;
            W2 = W2 - config.eta * dW2; b2 = b2 - config.eta * db2;
            W3 = W3 - config.eta * dW3; b3 = b3 - config.eta * db3;

          case 'momentum'
            V1 = config.mu * V1 + config.eta * dW1;
            V2 = config.mu * V2 + config.eta * dW2;
            V3 = config.mu * V3 + config.eta * dW3;
            W1 = W1 - V1; W2 = W2 - V2; W3 = W3 - V3;
            b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2; b3 = b3 - config.eta * db3;

          case 'nesterov'
            W1_look = W1 - config.mu * V1;
            W2_look = W2 - config.mu * V2;
            W3_look = W3 - config.mu * V3;
            % forward lookahead
            Z1_la = Xb * W1_look + b1; A1_la = forward(Z1_la, config.act1, config.leaky_alpha);
            Z2_la = A1_la * W2_look + b2; A2_la = forward(Z2_la, config.act2, config.leaky_alpha);
            Z3_la = A2_la * W3_look + b3; A3_la = softmax(Z3_la);
            dZ3_la = (A3_la - Yb)/B;
            dW3_la = A2_la' * dZ3_la; db3_la = sum(dZ3_la,1);
            dA2_la = dZ3_la * W3_look';
            dZ2_la = dA2_la .* backward(Z2_la, A2_la, config.act2, config.leaky_alpha);
            dW2_la = A1_la' * dZ2_la; db2_la = sum(dZ2_la,1);
            dA1_la = dZ2_la * W2_look';
            dZ1_la = dA1_la .* backward(Z1_la, A1_la, config.act1, config.leaky_alpha);
            dW1_la = Xb' * dZ1_la; db1_la = sum(dZ1_la,1);
            V1 = config.mu * V1 + config.eta * dW1_la;
            V2 = config.mu * V2 + config.eta * dW2_la;
            V3 = config.mu * V3 + config.eta * dW3_la;
            W1 = W1 - V1; W2 = W2 - V2; W3 = W3 - V3;
            b1 = b1 - config.eta * db1_la; b2 = b2 - config.eta * db2_la; b3 = b3 - config.eta * db3_la;

          case 'rmsprop'
            S1 = config.rho * S1 + (1-config.rho) * (dW1.^2);
            S2 = config.rho * S2 + (1-config.rho) * (dW2.^2);
            S3 = config.rho * S3 + (1-config.rho) * (dW3.^2);
            W1 = W1 - (config.eta ./ sqrt(S1 + config.eps_opt)) .* dW1;
            W2 = W2 - (config.eta ./ sqrt(S2 + config.eps_opt)) .* dW2;
            W3 = W3 - (config.eta ./ sqrt(S3 + config.eps_opt)) .* dW3;
            b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2; b3 = b3 - config.eta * db3;

          case 'adagrad'
            G1 = G1 + dW1.^2;
            G2 = G2 + dW2.^2;
            G3 = G3 + dW3.^2;
            W1 = W1 - (config.eta ./ sqrt(G1 + config.eps_opt)) .* dW1;
            W2 = W2 - (config.eta ./ sqrt(G2 + config.eps_opt)) .* dW2;
            W3 = W3 - (config.eta ./ sqrt(G3 + config.eps_opt)) .* dW3;
            b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2; b3 = b3 - config.eta * db3;

          case 'adam'
            t = t + 1;
            mW1 = config.beta1 * mW1 + (1-config.beta1) * dW1;
            mW2 = config.beta1 * mW2 + (1-config.beta1) * dW2;
            mW3 = config.beta1 * mW3 + (1-config.beta1) * dW3;
            vW1 = config.beta2 * vW1 + (1-config.beta2) * (dW1.^2);
            vW2 = config.beta2 * vW2 + (1-config.beta2) * (dW2.^2);
            vW3 = config.beta2 * vW3 + (1-config.beta2) * (dW3.^2);
            mW1_hat = mW1 / (1 - config.beta1^t);
            mW2_hat = mW2 / (1 - config.beta1^t);
            mW3_hat = mW3 / (1 - config.beta1^t);
            vW1_hat = vW1 / (1 - config.beta2^t);
            vW2_hat = vW2 / (1 - config.beta2^t);
            vW3_hat = vW3 / (1 - config.beta2^t);
            W1 = W1 - config.eta * mW1_hat ./ (sqrt(vW1_hat) + config.eps_opt);
            W2 = W2 - config.eta * mW2_hat ./ (sqrt(vW2_hat) + config.eps_opt);
            W3 = W3 - config.eta * mW3_hat ./ (sqrt(vW3_hat) + config.eps_opt);
            b1 = b1 - config.eta * db1; b2 = b2 - config.eta * db2; b3 = b3 - config.eta * db3;

          otherwise
            W1 = W1 - config.eta * dW1; b1 = b1 - config.eta * db1;
            W2 = W2 - config.eta * dW2; b2 = b2 - config.eta * db2;
            W3 = W3 - config.eta * dW3; b3 = b3 - config.eta * db3;
        end
      end % batches
    end % epochs

    % --- R2 train
    y_true_train = Train(:,end);
    A1_tr = forward(Xtrain * W1 + b1, config.act1, config.leaky_alpha);
    A2_tr = forward(A1_tr * W2 + b2, config.act2, config.leaky_alpha);
    A3_tr = softmax(A2_tr * W3 + b3);
    [~, pred_train] = max(A3_tr, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - SSres_train / SStot_train;
    if R2_train(r) < 0, R2_train(r) = 0; end

    % --- Test
    A1t = forward(Xtest * W1 + b1, config.act1, config.leaky_alpha);
    A2t = forward(A1t * W2 + b2, config.act2, config.leaky_alpha);
    A3t = softmax(A2t * W3 + b3);
    [~, pred] = max(A3t, [], 2);

    y_true_test = Test(:,end);
    y_pred_test = pred;
    TX_OK(r) = sum(pred == y_true_test) / size(Test,1) * 100;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - SSres_test / SStot_test;
    if R2_test(r) < 0, R2_test(r) = 0; end
  end % repeats

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('mlp2h: norm: %s, act1: %s, act2: %s, opt: %s, gd: %s, eta: %.4f, epochs: %d\n', ...
    config.normalization, config.act1, config.act2, config.opt_variant, config.gbds, config.eta, config.epochs);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5),R2_test_mean,  R2_train_mean, );

  rec_mean = mean(rec); prec_mean = mean(prec); f1_mean = mean(f1);
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
