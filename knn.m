% K-Nearest Neighbors classifier

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = knn(D, Nr, Ptrain, config)
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
  if ~isfield(config,'k'), config.k = 1; end

  for r = 1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % Normalization
    switch config.normalization
      case 'zscore'
        m = mean(Xtrain_raw,1); s = std(Xtrain_raw,0,1); s(s<epsn)=1;
        Xtrain = (Xtrain_raw - m) ./ s; Xtest = (Xtest_raw - m) ./ s;
      case 'minmax01'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng; Xtest = (Xtest_raw - mn) ./ rng;
      case 'minmax11'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn) ./ rng) - 1; Xtest = 2*((Xtest_raw - mn) ./ rng) - 1;
      otherwise
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
    end
    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % Distance computation
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
    % --- Compute R2 on training data
    Ytrain = Train(:,end);
    k = config.k;
    % distances for training
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
    % --- Compute R2 on test predictions
    y_true_test = Test(:,end);
    y_pred_test = preds;
    SSres_test = sum((y_true_test - y_pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - SSres_test / SStot_test;
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);

  fprintf('knn: normalization: %s, k: %d\n', config.normalization, config.k);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction