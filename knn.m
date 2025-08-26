% K-Nearest Neighbors classifier

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = knn(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none, robust, l2, whiten)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end
  if ~isfield(config,'k'), config.k = 1; end

  for r = 1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    switch lower(config.normalization)
      case 'zscore'
        m = mean(Xtrain_raw,1);
        s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s;
        Xtest  = (Xtest_raw  - m) ./ s;

      case 'minmax01'
        mn = min(Xtrain_raw,[],1);
        mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng;
        Xtest  = (Xtest_raw  - mn) ./ rng;

      case 'minmax11'
        mn = min(Xtrain_raw,[],1);
        mx = max(Xtrain_raw,[],1);
        rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn) ./ rng) - 1;
        Xtest  = 2*((Xtest_raw  - mn) ./ rng) - 1;

      case 'robust'
        med = median(Xtrain_raw,1);
        madv = median(abs(Xtrain_raw - med),1);
        % fator consistente para normal ~ 1.4826
        madv = madv * 1.4826;
        madv(madv<epsn)=1;
        Xtrain = (Xtrain_raw - med) ./ madv;
        Xtest  = (Xtest_raw  - med) ./ madv;

      case 'l2'
        % normaliza cada linha para norma 1
        ntrain = sqrt(sum(Xtrain_raw.^2,2)); ntrain(ntrain<epsn)=1;
        ntest  = sqrt(sum(Xtest_raw.^2,2));  ntest(ntest<epsn)=1;
        Xtrain = Xtrain_raw ./ ntrain;
        Xtest  = Xtest_raw  ./ ntest;

      case 'whiten'
        % centraliza
        m = mean(Xtrain_raw,1);
        Xc = bsxfun(@minus,Xtrain_raw,m);
        % covariância
        C = cov(Xc,1); % usa N em vez de N-1
        % regularização leve
        C = C + epsn*eye(size(C));
        % decomposição (eigen)
        [V, S] = eig(C);
        svals = diag(S);
        svals(svals<epsn)=epsn;
        W = V * diag(1./sqrt(svals)) * V';
        Xtrain = (Xc * W);
        Xtest  = (bsxfun(@minus,Xtest_raw,m) * W);

      case 'none'
        Xtrain = Xtrain_raw;
        Xtest  = Xtest_raw;

      otherwise
        error('Normalizacao desconhecida: %s', config.normalization);
    end

    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % Distâncias (euclidiana ao quadrado)
    sumT2 = sum(Xtest.^2, 2);
    sumX2 = sum(Xtrain.^2, 2)';
    D2 = bsxfun(@plus, sumT2, sumX2) - 2 * (Xtest * Xtrain');
    M = size(Xtest,1);
    preds = zeros(M,1);
    k = config.k;

    for i = 1:M
      [~, ix] = sort(D2(i,:), 'ascend');
      nn = ix(1:k);
      preds(i) = mode(Ytrain(nn));
    end

    TX_OK(r) = sum(preds == Ytest) / M * 100;

    % R2 treino
    sumT2_train = sum(Xtrain.^2,2);
    sumX2_train = sum(Xtrain.^2,2)';
    D2_train = bsxfun(@plus, sumT2_train, sumX2_train) - 2 * (Xtrain * Xtrain');
    Mtrain = size(Xtrain,1);
    preds_train = zeros(Mtrain,1);
    for i = 1:Mtrain
      [~, ix] = sort(D2_train(i,:), 'ascend');
      nn = ix(1:k);
      preds_train(i) = mode(Ytrain(nn));
    end
    SSres_train = sum((Ytrain - preds_train).^2);
    SStot_train = sum((Ytrain - mean(Ytrain)).^2);
    R2_train(r) = 1 - SSres_train / SStot_train;
    if R2_train(r) < 0, R2_train(r) = 0; end

    % R2 teste
    y_true_test = Ytest;
    y_pred_test = preds;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - SSres_test / SStot_test;
    if R2_test(r) < 0, R2_test(r) = 0; end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('knn: normalization: %s, k: %d\n', config.normalization, config.k);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
end