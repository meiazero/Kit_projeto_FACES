function [STATS, TX_OK] = nc(D, Nr, Ptrain, config)
  % Nearest Centroid classifier
  if nargin < 4, config = struct(); end
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
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
  end
  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  fprintf('nc: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5));
endfunction