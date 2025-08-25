% K-Nearest Neighbors classifier

function [STATS, TX_OK] = knn(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2 = zeros(Nr,1);
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

    % Coeficiente de determinação (R^2) entre rótulos e predições
    y_true = Test(:,end);
    y_pred = pred;
    SSres = sum((y_true - y_pred).^2);
    SStot = sum((y_true - mean(y_true)).^2);
    R2(r) = 1 - SSres / SStot;
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_mean = mean(R2);

  fprintf('knn: normalization: %s, k: %d\n', config.normalization, config.k);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f\n', STATS(1), STATS(2), STATS(3), STATS(4), STATS(5));
endfunction