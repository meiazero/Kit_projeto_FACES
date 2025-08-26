% Perceptron Logistic Classifier (com normalizacoes e otimizadores adicionais)
function [STATS, TX_OK, W, R2_train_mean, R2_test_mean, rec_mean, prec_mean, f1_mean] = pslog(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  rec = zeros(Nr,1);
  prec = zeros(Nr,1);
  f1 = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  % optization variants (gd, adam, adagrad, rmsprop)
  if ~isfield(config,'opt_variant'),  config.opt_variant  = 'gd'; end
  if ~isfield(config,'eta'),          config.eta = 0.01; end
  if ~isfield(config,'epochs'),       config.epochs = 200; end
  if ~isfield(config,'mu'),           config.mu = 0.9; end                % momentum / beta1
  if ~isfield(config,'rho'),          config.rho = 0.9; end               % rmsprop / adadelta rho
  if ~isfield(config,'beta1'),        config.beta1 = 0.9; end             % adam beta1
  if ~isfield(config,'beta2'),        config.beta2 = 0.999; end           % adam beta2
  if ~isfield(config,'eps_opt'),      config.eps_opt = 1e-8; end
  if ~isfield(config,'lambda'),       config.lambda = 0; end              % L2 regularization
  if ~isfield(config,'decay'),        config.decay = 0; end               % opcional: decaimento eta por epoca
  if ~isfield(config,'clip_grad'),    config.clip_grad = 0; end           % 0 = sem clipping, >0 = limite L2

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % Normalização
    switch lower(config.normalization)
      case {'zscore','standard'}
        m = mean(Xtrain_raw,1); s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s; Xtest = (Xtest_raw - m) ./ s;
      case 'minmax01'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng; Xtest = (Xtest_raw - mn) ./ rng;
      case 'minmax11'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn)./rng)-1; Xtest = 2*((Xtest_raw - mn)./rng)-1;
      case 'maxabs'
        mx = max(abs(Xtrain_raw),[],1); mx(mx<epsn)=1;
        Xtrain = Xtrain_raw ./ mx; Xtest = Xtest_raw ./ mx;
      case 'robust'   % median + IQR
        med = median(Xtrain_raw,1);
        q1 = quantile(Xtrain_raw,0.25,1);
        q3 = quantile(Xtrain_raw,0.75,1);
        iqrv = q3 - q1; iqrv(iqrv<epsn)=1;
        Xtrain = (Xtrain_raw - med) ./ iqrv; Xtest = (Xtest_raw - med) ./ iqrv;
      case 'l2feat'   % cada feature normalizada pelo seu L2 nos dados de treino
        nrm = sqrt(sum(Xtrain_raw.^2,1)); nrm(nrm<epsn)=1;
        Xtrain = Xtrain_raw ./ nrm; Xtest = Xtest_raw ./ nrm;
      case 'none'
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
      otherwise
        m = mean(Xtrain_raw,1); s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s; Xtest = (Xtest_raw - m) ./ s;
    end

    % One-hot
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain_oh(i, Train(i,end)) = 1; end

    Xb = [ones(Ntrain,1) Xtrain];
    % Inicialização Glorot-like
    W = randn(p+1, K) * sqrt(2/(p+K));

    % Estados otimizadores
    M = zeros(size(W));          % 1o momento (adam / nadam)
    V = zeros(size(W));          % 2o momento (adam / rmsprop)
    G2 = zeros(size(W));         % adagrad acumulador
    Delta = zeros(size(W));      % adadelta acumulador delta^2
    t = 0;                       % passo para bias correction

    for e=1:config.epochs
      t = t + 1;

      % forward
      Z = Xb * W;
      P = softmax(Z);

      % gradiente (log-loss)
      G = (Xb' * (P - Ytrain_oh)) / Ntrain;

      % regularização L2 (sem penalizar bias)
      if config.lambda > 0
        G = G + config.lambda * [zeros(1,K); W(2:end,:)];
      end

      % clipping (L2 global)
      if config.clip_grad > 0
        gnorm = sqrt(sum(G(:).^2));
        if gnorm > config.clip_grad
          G = G * (config.clip_grad / gnorm);
        end
      end

      % decaimento de taxa de aprendizado simples
      eta = config.eta / (1 + config.decay * (e-1));

      switch lower(config.opt_variant)
        case 'gd'
          W = W - eta * G;

        case 'momentum'
          M = config.mu * M + eta * G;
          W = W - M;

        case 'nesterov'
          % lookahead
          W_look = W - config.mu * M;
          P_la = softmax(Xb * W_look);
            G_la = (Xb' * (P_la - Ytrain_oh)) / Ntrain;
          if config.lambda > 0
            G_la = G_la + config.lambda * [zeros(1,K); W_look(2:end,:)];
          end
          M = config.mu * M + eta * G_la;
          W = W - M;

        case 'adagrad'
          G2 = G2 + G.^2;
          W = W - (eta ./ sqrt(G2 + config.eps_opt)) .* G;

        case 'rmsprop'
          V = config.rho * V + (1 - config.rho) * (G.^2);
          W = W - (eta ./ (sqrt(V) + config.eps_opt)) .* G;

        case 'adadelta'
          V = config.rho * V + (1 - config.rho) * (G.^2);
          update = sqrt(Delta + config.eps_opt) ./ sqrt(V + config.eps_opt) .* G;
          W = W - update;
          Delta = config.rho * Delta + (1 - config.rho) * (update.^2);

        case 'adam'
          M = config.beta1 * M + (1 - config.beta1) * G;
          V = config.beta2 * V + (1 - config.beta2) * (G.^2);
          Mhat = M / (1 - config.beta1^t);
          Vhat = V / (1 - config.beta2^t);
          W = W - eta * Mhat ./ (sqrt(Vhat) + config.eps_opt);

        case 'adamw'
          M = config.beta1 * M + (1 - config.beta1) * G;
          V = config.beta2 * V + (1 - config.beta2) * (G.^2);
          Mhat = M / (1 - config.beta1^t);
          Vhat = V / (1 - config.beta2^t);
          % decoupled weight decay (não no bias)
          W(2:end,:) = W(2:end,:) - eta * config.lambda * W(2:end,:);
          W = W - eta * Mhat ./ (sqrt(Vhat) + config.eps_opt);

        case 'nadam'
          M = config.beta1 * M + (1 - config.beta1) * G;
          V = config.beta2 * V + (1 - config.beta2) * (G.^2);
          Mhat = M / (1 - config.beta1^t);
          Vhat = V / (1 - config.beta2^t);
          M_nesterov = (config.beta1 * Mhat) + ((1 - config.beta1) * G) / (1 - config.beta1^t);
          W = W - eta * M_nesterov ./ (sqrt(Vhat) + config.eps_opt);

        otherwise
          W = W - eta * G;
      end
    end

    % R2 treino (apenas ilustrativo; não muito significativo para classificação)
    y_true_train = Train(:,end);
    Ztrain = Xb * W;
    [~, pred_train] = max(Ztrain, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - SSres_train / max(SStot_train, epsn);
    if R2_train(r) < 0, R2_train(r) = 0; end

    % Teste
    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ztest = Xtest_b * W;
    [~, pred] = max(Ztest, [], 2);

    y_true_test = Test(:,end);
    [rec(r), prec(r), f1(r)] = classification_metrics(y_true_test, pred);
    TX_OK(r) = mean(pred == y_true_test) * 100;

    SSres_test = sum((y_true_test - pred).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - SSres_test / max(SStot_test, epsn);
    if R2_test(r) < 0, R2_test(r) = 0; end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);
  rec_mean = mean(rec);
  prec_mean = mean(prec);
  f1_mean = mean(f1);

  fprintf('pslog: norm=%s, opt=%s, eta=%.4f, epochs=%d\n', ...
    config.normalization, config.opt_variant, config.eta, config.epochs);
  fprintf('Stats - acc_mean: %.3f, min: %.3f, max: %.3f, med: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction

function S = softmax(Z)
  Zs = Z - max(Z,[],2);
  EZ = exp(Zs);
  S = EZ ./ sum(EZ,2);
endfunction
