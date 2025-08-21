function [STATS TX_OK W] = pslog(D, Nr, Ptrain)
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

    Xtrain = (Xtrain_raw - m) ./ s;
    Xtest_raw = Test(:,1:p);
    Xtest = (Xtest_raw - m) ./ s;
    Ytrain_oh = zeros(Ntrain, K);

    for i=1:Ntrain
      Ytrain_oh(i, Train(i,end)) = 1;
    end

    W = rand(p+1, K) * 0.1;
    Xtrain_b = [ones(Ntrain,1) Xtrain];
    eta = 0.01; epochs = 200;

    for e=1:epochs
      Z = Xtrain_b * W;
      P = softmax(Z')';
      grad = Xtrain_b' * (P - Ytrain_oh);
      W = W - eta * grad;
    end

    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ztest = Xtest_b * W;
    [~, pred] = max(Ztest, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;
  end
  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction

function s = softmax(x)
  ex = exp(x - max(x,[],2));
  s = ex ./ sum(ex,2);
endfunction