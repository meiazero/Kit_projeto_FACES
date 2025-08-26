% Linear Discriminant Analysis classifier (multi-class)

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = lda(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none, robust, l2, whiten)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end

  for r = 1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % -------- Normalizações adicionadas --------
    switch lower(config.normalization)
      case 'zscore'                 % (x - mean)/std (amostral)
        m = mean(Xtrain_raw,1);
        s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s;
        Xtest  = (Xtest_raw  - m) ./ s;

      case 'zscore-pop'             % (x - mean)/std populacional
        m = mean(Xtrain_raw,1);
        s = std(Xtrain_raw,1,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s;
        Xtest  = (Xtest_raw  - m) ./ s;

      case 'center'                 % centralização apenas
        m = mean(Xtrain_raw,1);
        Xtrain = Xtrain_raw - m;
        Xtest  = Xtest_raw  - m;

      case 'minmax01'               % [0,1]
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng;
        Xtest  = (Xtest_raw  - mn) ./ rng;

      case 'minmax11'               % [-1,1]
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn)./rng) - 1;
        Xtest  = 2*((Xtest_raw  - mn)./rng) - 1;

      case 'maxabs'                 % divide pelo |max| de cada atributo
        mx = max(abs(Xtrain_raw),[],1); mx(mx<epsn)=1;
        Xtrain = Xtrain_raw ./ mx;
        Xtest  = Xtest_raw  ./ mx;

      case 'robust'                 % (x - mediana)/IQR
        med = median(Xtrain_raw,1);
        q1 = quantile(Xtrain_raw,0.25,1);
        q3 = quantile(Xtrain_raw,0.75,1);
        iqrv = q3 - q1; iqrv(iqrv<epsn)=1;
        Xtrain = (Xtrain_raw - med) ./ iqrv;
        Xtest  = (Xtest_raw  - med) ./ iqrv;

      case 'l2'                     % normalização L2 por amostra (linha)
        ntr = sqrt(sum(Xtrain_raw.^2,2)); ntr(ntr<epsn)=1;
        nts = sqrt(sum(Xtest_raw.^2,2));  nts(nts<epsn)=1;
        Xtrain = Xtrain_raw ./ ntr;
        Xtest  = Xtest_raw  ./ nts;

      case 'none'
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;

      otherwise
        warning('Normalização desconhecida: %s. Usando nenhuma.', config.normalization);
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
    end
    % -------------------------------------------

    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % Médias e priors
    mu = zeros(K, p);
    prior = zeros(K,1);
    for c = 1:K
      Xc = Xtrain(Ytrain==c, :);
      nc = size(Xc,1);
      prior(c) = nc / Ntrain;
      mu(c,:) = mean(Xc,1);
    end

    % Covariância intra-classes
    Sigma = zeros(p,p);
    for c = 1:K
      Xc = Xtrain(Ytrain==c, :);
      Xc_centered = Xc - mu(c,:);
      Sigma = Sigma + Xc_centered' * Xc_centered;
    end
    Sigma = Sigma / (Ntrain - K);
    Sigma = Sigma + epsn * eye(p);
    invSigma = inv(Sigma);

    % Classificação teste
    M = size(Xtest,1);
    preds = zeros(M,1);
    for i = 1:M
      x = Xtest(i,:);
      scores = zeros(K,1);
      for c = 1:K
        scores(c) = x * invSigma * mu(c,:)' - 0.5 * mu(c,:) * invSigma * mu(c,:)' + log(prior(c) + epsn);
      end
      [~, preds(i)] = max(scores);
    end
    TX_OK(r) = sum(preds == Ytest) / M * 100;

    % R2 train
    Ntrain_loc = size(Xtrain,1);
    preds_train = zeros(Ntrain_loc,1);
    for i = 1:Ntrain_loc
      x_tr = Xtrain(i,:);
      scores_tr = zeros(K,1);
      for c = 1:K
        scores_tr(c) = x_tr * invSigma * mu(c,:)' - 0.5 * mu(c,:) * invSigma * mu(c,:)' + log(prior(c) + epsn);
      end
      [~, preds_train(i)] = max(scores_tr);
    end
    SSres_train = sum((Ytrain - preds_train).^2);
    SStot_train = sum((Ytrain - mean(Ytrain)).^2);
    R2_train(r) = 1 - SSres_train / SStot_train;
    if R2_train(r) < 0, R2_train(r)=0; end

    % R2 teste
    y_true_test = Ytest;
    y_pred_test = preds;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - SSres_test / SStot_test;
    if R2_test(r) < 0, R2_test(r)=0; end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('lda: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
end