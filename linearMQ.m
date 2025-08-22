function [STATS TX_OK W] = linearMQ(D, Nr, Ptrain, config)
  % linearMQ com config opcional (struct)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  epsn = 1e-8;

  % defaults
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % Normalizacao conforme config
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
        Xtrain = 2 * ((Xtrain_raw - mn) ./ rng) - 1;
        Xtest  = 2 * ((Xtest_raw  - mn) ./ rng) - 1;
      case 'none'
        Xtrain = Xtrain_raw; Xtest = Xtest_raw;
      otherwise
        Xtrain = (Xtrain_raw - mean(Xtrain_raw,1)) ./ max(std(Xtrain_raw,0,1), epsn);
        Xtest  = (Xtest_raw  - mean(Xtrain_raw,1)) ./ max(std(Xtrain_raw,0,1), epsn);
    end

    % One-hot
    Ytrain = zeros(Ntrain, K);
    for i=1:Ntrain, Ytrain(i, Train(i,end)) = 1; end

    % Pseudoinversa com bias
    Xtrain_b = [ones(Ntrain,1) Xtrain];
    W = pinv(Xtrain_b) * Ytrain;

    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ypred = Xtest_b * W;
    [~, pred] = max(Ypred, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];

  fprintf('linearMQ: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f\n', STATS(1), STATS(2), STATS(3), STATS(4), STATS(5));
endfunction
