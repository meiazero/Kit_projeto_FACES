% MLP 1-hidden layer classifier

function [STATS, TX_OK, W1, W2, R2_train_mean, R2_test_mean, rec_mean, prec_mean, f1_mean] = mlp1h(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H = 16;

  TX_OK    = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test  = zeros(Nr,1);
  rec  = zeros(Nr,1); prec = zeros(Nr,1); f1 = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  % hidden activation variants (sigmoid, tanh, relu, leakyrelu)
  if ~isfield(config,'act1'),          config.act1 = 'sigmoid'; end
  % optization variants (gd, adam, adagrad, rmsprop)
  if ~isfield(config,'opt_variant'),   config.opt_variant = 'gd'; end
  % gradient descent strategy (sgd, batch-gd, mini-batch-gd)
  if ~isfield(config,'gbds'),          config.gbds = 'batch-gd'; end

  if ~isfield(config,'batch_size'),    config.batch_size = 16; end
  if ~isfield(config,'eta'),           config.eta = 0.003; end
  if ~isfield(config,'epochs'),        config.epochs = 1; end
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

    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain_oh(i, Train(i,end)) = 1; end

    if any(strcmp(config.act1, {'relu','leakyrelu'}))
      W1 = randn(p, H) * sqrt(2/p);
    else
      W1 = randn(p, H) * sqrt(1/p);
    end
    b1 = zeros(1, H);
    W2 = randn(H, K) * sqrt(1/H);
    b2 = zeros(1, K);

    V1 = zeros(size(W1)); V2 = zeros(size(W2));
    S1 = zeros(size(W1)); S2 = zeros(size(W2));

    for e=1:config.epochs
      % Para otimizadores que usam gradiente em lote precisamos do forward completo
      Z1_full = Xtrain * W1 + b1;
      A1_full = forward(Z1_full, config.act1, config.leaky_alpha);
      Z2_full = A1_full * W2 + b2;
      A2_full = softmax(Z2_full);
      dZ2_full = (A2_full - Ytrain_oh) / Ntrain;
      dW2_full = A1_full' * dZ2_full; db2_full = sum(dZ2_full,1);
      dZ1_full = (dZ2_full * W2') .* backward(Z1_full, A1_full, config.act1, config.leaky_alpha);
      dW1_full = Xtrain' * dZ1_full; db1_full = sum(dZ1_full,1);

      switch config.opt_variant
        case 'gd'
          switch config.gbds
            case 'batch-gd'
              % Usa gradiente completo
              V1 = config.mu * V1 - config.eta * dW1_full;
              V2 = config.mu * V2 - config.eta * dW2_full;
              W1 = W1 + V1; W2 = W2 + V2;
              b1 = b1 - config.eta * db1_full;
              b2 = b2 - config.eta * db2_full;

            case 'sgd'
              order = randperm(Ntrain);
              for ii = 1:Ntrain
                k = order(ii);
                xk = Xtrain(k,:); yk = Ytrain_oh(k,:);
                z1 = xk * W1 + b1;
                a1 = forward(z1, config.act1, config.leaky_alpha);
                z2 = a1 * W2 + b2;
                a2 = softmax(z2);
                dz2 = (a2 - yk);             % 1 x K
                dw2 = a1' * dz2;             % H x K
                db2 = dz2;
                dz1 = (dz2 * W2') .* backward(z1, a1, config.act1, config.leaky_alpha); % 1 x H
                dw1 = xk' * dz1;             % p x H
                db1 = dz1;

                V1 = config.mu * V1 - config.eta * dw1;
                V2 = config.mu * V2 - config.eta * dw2;
                W1 = W1 + V1; W2 = W2 + V2;
                b1 = b1 - config.eta * db1;
                b2 = b2 - config.eta * db2;
              end

            case 'mini-batch-gd'
              B = config.batch_size;
              order = randperm(Ntrain);
              for start = 1:B:Ntrain
                stop = min(start+B-1, Ntrain);
                ids = order(start:stop);
                Xb = Xtrain(ids,:); Yb = Ytrain_oh(ids,:);
                Nb = size(Xb,1);

                Z1b = Xb * W1 + b1;
                A1b = forward(Z1b, config.act1, config.leaky_alpha);
                Z2b = A1b * W2 + b2;
                A2b = softmax(Z2b);

                dZ2b = (A2b - Yb) / Nb;
                dW2 = A1b' * dZ2b; db2 = sum(dZ2b,1);
                dZ1b = (dZ2b * W2') .* backward(Z1b, A1b, config.act1, config.leaky_alpha);
                dW1 = Xb' * dZ1b; db1 = sum(dZ1b,1);

                V1 = config.mu * V1 - config.eta * dW1;
                V2 = config.mu * V2 - config.eta * dW2;
                W1 = W1 + V1; W2 = W2 + V2;
                b1 = b1 - config.eta * db1;
                b2 = b2 - config.eta * db2;
              end
            otherwise
              error('Valor inválido em config.gbds (use sgd, batch-gd, mini-batch-gd)');
          end

        case 'adam'
          beta1 = 0.9; beta2 = 0.999;
          V1 = beta1 * V1 + (1-beta1) * dW1_full;
          V2 = beta1 * V2 + (1-beta1) * dW2_full;
          S1 = beta2 * S1 + (1-beta2) * dW1_full.^2;
            S2 = beta2 * S2 + (1-beta2) * dW2_full.^2;
          V1c = V1 / (1-beta1^e); V2c = V2 / (1-beta1^e);
          S1c = S1 / (1-beta2^e); S2c = S2 / (1-beta2^e);
          W1 = W1 - config.eta * V1c ./ (sqrt(S1c) + config.eps_opt);
          W2 = W2 - config.eta * V2c ./ (sqrt(S2c) + config.eps_opt);
          b1 = b1 - config.eta * db1_full;
          b2 = b2 - config.eta * db2_full;

        case 'adagrad'
          S1 = S1 + dW1_full.^2;
          S2 = S2 + dW2_full.^2;
          W1 = W1 - config.eta * dW1_full ./ (sqrt(S1) + config.eps_opt);
          W2 = W2 - config.eta * dW2_full ./ (sqrt(S2) + config.eps_opt);
          b1 = b1 - config.eta * db1_full;
          b2 = b2 - config.eta * db2_full;

        case 'rmsprop'
          S1 = config.rho * S1 + (1-config.rho) * dW1_full.^2;
          S2 = config.rho * S2 + (1-config.rho) * dW2_full.^2;
          W1 = W1 - config.eta * dW1_full ./ (sqrt(S1) + config.eps_opt);
          W2 = W2 - config.eta * dW2_full ./ (sqrt(S2) + config.eps_opt);
          b1 = b1 - config.eta * db1_full;
          b2 = b2 - config.eta * db2_full;
      end
    end % epochs

    % R2 treino
    y_true_train = Train(:,end);
    Z1_tr = Xtrain * W1 + b1;
    A1_tr = forward(Z1_tr, config.act1, config.leaky_alpha);
    Z2_tr = A1_tr * W2 + b2;
    A2_tr = softmax(Z2_tr);
    [~, pred_train] = max(A2_tr, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - (SSres_train / SStot_train);
    if R2_train(r) < 0, R2_train(r) = 0; end

    % Teste
    Z1t = Xtest * W1 + b1;
    A1t = forward(Z1t, config.act1, config.leaky_alpha);
    Z2t = A1t * W2 + b2;
    A2t = softmax(Z2t);
    [~, pred] = max(A2t, [], 2);
    y_true_test = Test(:,end);
    y_pred_test = pred;
    TX_OK(r) = sum(pred == y_true_test) / size(Test,1) * 100;

    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - (SSres_test / SStot_test);
    if R2_test(r) < 0, R2_test(r) = 0; end
  end % repeats

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('mlp1h: norm: %s, act1: %s, opt: %s, gd: %s, eta: %.4f, epochs: %d, batch_size: %d\n', ...
    config.normalization, config.act1, config.opt_variant, config.gbds, config.eta, config.epochs, config.batch_size);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f,  R2_train: %.3f\n',  ...
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
