function [STATS TX_OK W] = linearMQ(D, Nr, Ptrain)
  [N, p1] = size(D); p = p1 - 1; K = max(D(:,end));
  TX_OK = zeros(Nr,1);

  epsn = 1e-8;

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw = Test(:,1:p);

    % Sem normalizacao
    % Xtrain = Xtrain_raw;
    % Xtest  = Xtest_raw;

    % Normalize features (z-score)
    m = mean(Xtrain_raw, 1);
    s = std(Xtrain_raw, 0, 1);
    s(s<epsn)=1;
    Xtrain = (Xtrain_raw - m)./(s);
    Xtest  = (Xtest_raw - m)./(s);

    % Normalize features (mudanca de escala [-1,+1])
    % mn = min(Xtrain_raw,[],1);
    % mx = max(Xtrain_raw,[],1);
    % rng = mx - mn;
    % rng(rng<epsn)=1;
    % Xtrain = (Xtrain_raw - mn)./rng;
    % Xtest  = (Xtest_raw - mn)./rng;

    % Normalize features (mudanca de escala [0,+1])
    % mn = min(Xtrain_raw,[],1);
    % mx = max(Xtrain_raw,[],1);
    % rng = mx - mn;
    % rng(rng<epsn)=1;
    % Xtrain = 2*((Xtrain_raw - mn)./rng) - 1;
    % Xtest  = 2*((Xtest_raw - mn)./rng) - 1;

    % One-hot
    Ytrain = zeros(Ntrain, K);
    for i=1:Ntrain
      Ytrain(i, Train(i,end)) = 1;
    end

    % Compute weights using pseudoinverse
    Xtrain_b = [ones(Ntrain,1) Xtrain];
    W = pinv(Xtrain_b) * Ytrain;

    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ypred = Xtest_b * W;

    [~, pred] = max(Ypred, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;
  end
  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction
