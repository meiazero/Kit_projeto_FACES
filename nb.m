% Naive Bayes classifier

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = nb(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D); p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test  = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none, robustz, l2, mean0)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end

  for r = 1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % --- Normalization ---
    [Xtrain, Xtest] = normalization(Xtrain_raw, Xtest_raw, config.normalization, epsn);

    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % --- Parameter estimation (Gaussian NB)
    mu = zeros(K,p); sigma2 = zeros(K,p); prior = zeros(K,1);
    for c = 1:K
      Xc = Xtrain(Ytrain==c,:);
      nc = size(Xc,1);
      prior(c) = nc / Ntrain;
      mu(c,:) = mean(Xc,1);
      sigma2(c,:) = var(Xc,0,1) + epsn;
    end

    % --- Predict test
    preds = classify(Xtest, mu, sigma2, prior, epsn);

    TX_OK(r) = mean(preds == Ytest) * 100;

    % --- R2 train
    preds_train = classify(Xtrain, mu, sigma2, prior, epsn);
    R2_train(r) = max(0, 1 - sum((Ytrain - preds_train).^2) / sum((Ytrain - mean(Ytrain)).^2));

    % --- R2 test
    R2_test(r) = max(0, 1 - sum((Ytest - preds).^2) / sum((Ytest - mean(Ytest)).^2));
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('nb: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
end

function [Xtrain, Xtest] = normalization(Xtr, Xte, method, epsn)
  switch lower(method)
    case 'zscore'
      m = mean(Xtr,1); s = std(Xtr,0,1); s(s<epsn)=1;
      Xtrain = (Xtr - m)./s; Xtest = (Xte - m)./s;
    case 'minmax01'
      mn = min(Xtr,[],1); mx = max(Xtr,[],1); rng = mx - mn; rng(rng<epsn)=1;
      Xtrain = (Xtr - mn)./rng; Xtest = (Xte - mn)./rng;
    case 'minmax11'
      mn = min(Xtr,[],1); mx = max(Xtr,[],1); rng = mx - mn; rng(rng<epsn)=1;
      Xtrain = 2*((Xtr - mn)./rng) - 1; Xtest = 2*((Xte - mn)./rng) - 1;
    case 'l2'
      ntr = sqrt(sum(Xtr.^2,2)); ntr(ntr<epsn)=1;
      nte = sqrt(sum(Xte.^2,2)); nte(nte<epsn)=1;
      Xtrain = Xtr ./ ntr; Xtest = Xte ./ nte;
    case 'mean0'
      m = mean(Xtr,1);
      Xtrain = Xtr - m; Xtest = Xte - m;
    case 'robustz'
      med = median(Xtr,1);
      q1 = quantile(Xtr,0.25,1); q3 = quantile(Xtr,0.75,1);
      iqrv = q3 - q1; iqrv(iqrv<epsn)=1;
      Xtrain = (Xtr - med)./iqrv; Xtest = (Xte - med)./iqrv;
    case 'none'
      Xtrain = Xtr; Xtest = Xte;
    otherwise
      error('Normalizacao desconhecida: %s', method);
  end
end

function preds = classify(X, mu, sigma2, prior, epsn)
  K = size(mu,1);
  M = size(X,1);
  preds = zeros(M,1);
  log2pi = log(2*pi);
  for i = 1:M
    x = X(i,:);
    logp = zeros(K,1);
    for c = 1:K
      logp(c) = -0.5 * sum(log2pi + log(sigma2(c,:))) ...
                -0.5 * sum(((x - mu(c,:)).^2) ./ sigma2(c,:)) ...
                + log(prior(c) + epsn);
    end
    [~, preds(i)] = max(logp);
  end
end