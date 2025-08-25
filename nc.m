% Nearest Centroid classifier

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = nc(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test = zeros(Nr,1);
  epsn = 1e-8;

  % default normalization
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end

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
        Xtrain = (Xtrain_raw - m) ./ s;
        Xtest  = (Xtest_raw  - m) ./ s;
      case 'minmax01'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = (Xtrain_raw - mn) ./ rng;
        Xtest  = (Xtest_raw  - mn) ./ rng;
      case 'minmax11'
        mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1); rng = mx - mn; rng(rng<epsn)=1;
        Xtrain = 2*((Xtrain_raw - mn)./rng) - 1;
        Xtest  = 2*((Xtest_raw  - mn)./rng) - 1;
      otherwise
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
    end
    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % Compute centroids
    centroids = zeros(K, p);
    for c = 1:K
      Xc = Xtrain(Ytrain==c, :);
      if isempty(Xc)
        centroids(c,:) = zeros(1,p);
      else
        centroids(c,:) = mean(Xc,1);
      end
    end

    % Classify
    M = size(Xtest,1);
    preds = zeros(M,1);
    for i = 1:M
      x = Xtest(i,:);

      % Euclidean distances to centroids
      d2 = sum((centroids - x).^2, 2);
      [~, preds(i)] = min(d2);
    end

    TX_OK(r) = sum(preds == Ytest) / M * 100;
    % --- Compute R2 on training data
    Ytrain = Train(:,end);
    Ntrain = size(Xtrain,1);
    preds_train = zeros(Ntrain,1);
    for i = 1:Ntrain
      x_tr = Xtrain(i,:);
      d2_tr = sum((centroids - x_tr).^2, 2);
      [~, preds_train(i)] = min(d2_tr);
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

    % Coeficiente de determinação (R^2) entre rótulos e predições
    y_true = Test(:,end);
    y_pred = pred;
    SSres = sum((y_true - y_pred).^2);
    SStot = sum((y_true - mean(y_true)).^2);
    R2(r) = 1 - SSres / SStot;
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);

  fprintf('nc: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction