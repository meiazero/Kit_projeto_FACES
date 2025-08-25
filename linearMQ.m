% Linear MQ classifier

function [STATS, TX_OK, W, R2_train_mean, R2_test_mean, rec_mean, prec_mean, f1_mean] = linearMQ(D, Nr, Ptrain, config)
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

    % Solve linear regression via least squares (with bias)
    Xtrain_b = [ones(Ntrain,1) Xtrain];
    W = Xtrain_b \ Ytrain;

    % --- Compute R2 on training data
    y_true_train = Train(:,end);
    Xtrain_b = [ones(size(Xtrain,1),1) Xtrain];
    Ypred_train = Xtrain_b * W;
    [~, pred_train] = max(Ypred_train, [], 2);
    SSres_train = sum((y_true_train - pred_train).^2);
    SStot_train = sum((y_true_train - mean(y_true_train)).^2);
    R2_train(r) = 1 - (SSres_train / SStot_train);
    if R2_train(r) < 0, R2_train(r) = 0; end

    % --- Test predictions and R2
    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ypred_test = Xtest_b * W;
    [~, pred_test] = max(Ypred_test, [], 2);
    % classification metrics for test
    y_true_test = Test(:,end);
    [rec(r), prec(r), f1(r)] = classification_metrics(y_true_test, pred_test);
    % accuracy
    correct = sum(pred_test == y_true_test);
    TX_OK(r) = correct / size(Test,1) * 100;
    % R2 for test
    SSres_test = sum((y_true_test - pred_test).^2);
    SStot_test = sum((y_true_test - mean(y_true_test)).^2);
    R2_test(r) = 1 - (SSres_test / SStot_test);
    if R2_test(r) < 0, R2_test(r) = 0; end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean = mean(R2_test);
  % summary of classification metrics
  rec_mean = mean(rec);
  prec_mean = mean(prec);
  f1_mean = mean(f1);

  fprintf('linearMQ: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n',...
   STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction
