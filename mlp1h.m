function [STATS TX_OK W1 W2] = mlp1h(D, Nr, Ptrain)
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H = 20;
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

    W1 = randn(p, H) * sqrt(2/p);
    b1 = zeros(1, H);
    W2 = randn(H, K) * sqrt(2/H);
    b2 = zeros(1, K);
    eta = 0.01; epochs = 100;

    for e=1:epochs
      Z1 = Xtrain * W1 + b1;
      A1 = 1 ./ (1 + exp(-Z1));
      Z2 = A1 * W2 + b2;
      A2 = softmax(Z2')';
      dZ2 = A2 - Ytrain_oh;
      dW2 = A1' * dZ2 / Ntrain;
      db2 = sum(dZ2) / Ntrain;
      dA1 = dZ2 * W2';
      dZ1 = dA1 .* A1 .* (1 - A1);
      dW1 = Xtrain' * dZ1 / Ntrain;
      db1 = sum(dZ1) / Ntrain;
      W1 = W1 - eta * dW1;
      b1 = b1 - eta * db1;
      W2 = W2 - eta * dW2;
      b2 = b2 - eta * db2;
    end

    Z1_test = Xtest * W1 + b1;
    A1_test = 1 ./ (1 + exp(-Z1_test));
    Z2_test = A1_test * W2 + b2;
    A2_test = softmax(Z2_test')';
    [~, pred] = max(A2_test, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;
  end
  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction

function s = softmax(x)
  ex = exp(x - max(x,[],2));
  s = ex ./ sum(ex,2);
endfunction