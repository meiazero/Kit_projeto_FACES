function [STATS TX_OK W] = linearMQ(D, Nr, Ptrain)
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);

    m = mean(Xtrain_raw);
    s = std(Xtrain_raw);

    Xtrain_norm = (Xtrain_raw - m) ./ s;
    Xtrain = [ones(Ntrain,1) Xtrain_norm];
    Ytrain = zeros(Ntrain, K);


    for i=1:Ntrain
      Ytrain(i, Train(i,end)) = 1;
    end

    W = pinv(Xtrain) * Ytrain;
    Xtest_raw = Test(:,1:p);
    Xtest_norm = (Xtest_raw - m) ./ s;
    Xtest = [ones(size(Test,1),1) Xtest_norm];
    Ypred = Xtest * W;
    [~, pred] = max(Ypred, [], 2);

    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;

    disp(["<MQ, round = " num2str(r) "/" num2str(Nr) ", mean = " num2str(m) ", std = " num2str(s) ">"]);
  end
  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction
