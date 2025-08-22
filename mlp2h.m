function [STATS TX_OK W1 W2 W3] = mlp2h(D, Nr, Ptrain)
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H1 = 20;
  H2 = 10;
  TX_OK = zeros(Nr,1);

  epsn = 1e-8;

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    % -----------------------------
    % Normalização (descomente uma)
    % -----------------------------
    % Z-score (padrão)
    m = mean(Xtrain_raw, 1);
    s = std(Xtrain_raw, 0, 1);
    s(s<epsn)=1;
    Xtrain = (Xtrain_raw - m)./s;
    Xtest  = (Xtest_raw - m)./s;

    % MinMax [0,+1]
    % mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
    % rng = mx - mn; rng(rng<epsn)=1;
    % Xtrain = (Xtrain_raw - mn)./rng;
    % Xtest  = (Xtest_raw - mn)./rng;

    % MinMax [-1,+1]
    % mn = min(Xtrain_raw,[],1); mx = max(Xtrain_raw,[],1);
    % rng = mx - mn; rng(rng<epsn)=1;
    % Xtrain = 2*((Xtrain_raw - mn)./rng) - 1;
    % Xtest  = 2*((Xtest_raw - mn)./rng) - 1;

    % Sem normalização
    % Xtrain = Xtrain_raw;
    % Xtest  = Xtest_raw;

    % -----------------------------
    % One-hot
    % -----------------------------
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain
      Ytrain_oh(i, Train(i,end)) = 1;
    end

    % -----------------------------
    % Configurações (descomente/ajuste)
    % -----------------------------
    act1 = 'sigmoid';   % 'sigmoid'|'tanh'|'relu'|'leakyrelu'
    act2 = 'sigmoid';   % mesma escolha para 2a camada oculta
    eta = 0.01;
    epochs = 200;
    opt_variant = 'gd'; % 'gd'|'momentum'|'nesterov'|'rmsprop'
    mu = 0.9; rho = 0.9; eps_opt = 1e-8; leaky_alpha = 0.01;

    % -----------------------------
    % Inicialização
    % -----------------------------
    if any(strcmp(act1, {'relu','leakyrelu'})), W1 = randn(p, H1) * sqrt(2/p);
    else,                                        W1 = randn(p, H1) * sqrt(1/p);
    end
    b1 = zeros(1, H1);

    if any(strcmp(act2, {'relu','leakyrelu'})), W2 = randn(H1, H2) * sqrt(2/H1);
    else,                                        W2 = randn(H1, H2) * sqrt(1/H1);
    end
    b2 = zeros(1, H2);

    W3 = randn(H2, K) * sqrt(1/H2);
    b3 = zeros(1, K);

    V1 = zeros(size(W1)); V2 = zeros(size(W2)); V3 = zeros(size(W3));
    S1 = zeros(size(W1)); S2 = zeros(size(W2)); S3 = zeros(size(W3));

    % -----------------------------
    % Treinamento
    % -----------------------------
    for e=1:epochs
      % Forward
      Z1 = Xtrain * W1 + b1;
      A1 = act_forward(Z1, act1, leaky_alpha);

      Z2 = A1 * W2 + b2;
      A2 = act_forward(Z2, act2, leaky_alpha);

      Z3 = A2 * W3 + b3;
      A3 = softmax_rows(Z3);

      % Gradientes (média)
      dZ3 = (A3 - Ytrain_oh) / Ntrain;
      dW3 = A2' * dZ3; db3 = sum(dZ3,1);

      dA2 = dZ3 * W3';
      dZ2 = dA2 .* act_backward(Z2, A2, act2, leaky_alpha);
      dW2 = A1' * dZ2; db2 = sum(dZ2,1);

      dA1 = dZ2 * W2';
      dZ1 = dA1 .* act_backward(Z1, A1, act1, leaky_alpha);
      dW1 = Xtrain' * dZ1; db1 = sum(dZ1,1);

      % Atualização
      switch opt_variant
        case 'gd'
          W1 = W1 - eta*dW1; b1 = b1 - eta*db1;
          W2 = W2 - eta*dW2; b2 = b2 - eta*db2;
          W3 = W3 - eta*dW3; b3 = b3 - eta*db3;

        case 'momentum'
          V1 = mu*V1 + eta*dW1; V2 = mu*V2 + eta*dW2; V3 = mu*V3 + eta*dW3;
          W1 = W1 - V1; W2 = W2 - V2; W3 = W3 - V3;
          b1 = b1 - eta*db1; b2 = b2 - eta*db2; b3 = b3 - eta*db3;

        case 'nesterov'
          W1_look = W1 - mu*V1; W2_look = W2 - mu*V2; W3_look = W3 - mu*V3;
          Z1_la = Xtrain*W1_look + b1; A1_la = act_forward(Z1_la, act1, leaky_alpha);
          Z2_la = A1_la*W2_look + b2; A2_la = act_forward(Z2_la, act2, leaky_alpha);
          Z3_la = A2_la*W3_look + b3; A3_la = softmax_rows(Z3_la);

          dZ3_la = (A3_la - Ytrain_oh)/Ntrain;
          dW3_la = A2_la' * dZ3_la; db3_la = sum(dZ3_la,1);
          dA2_la = dZ3_la * W3_look';
          dZ2_la = dA2_la .* act_backward(Z2_la, A2_la, act2, leaky_alpha);
          dW2_la = A1_la' * dZ2_la; db2_la = sum(dZ2_la,1);
          dA1_la = dZ2_la * W2_look';
          dZ1_la = dA1_la .* act_backward(Z1_la, A1_la, act1, leaky_alpha);
          dW1_la = Xtrain' * dZ1_la; db1_la = sum(dZ1_la,1);

          V1 = mu*V1 + eta*dW1_la; V2 = mu*V2 + eta*dW2_la; V3 = mu*V3 + eta*dW3_la;
          W1 = W1 - V1; W2 = W2 - V2; W3 = W3 - V3;
          b1 = b1 - eta*db1_la; b2 = b2 - eta*db2_la; b3 = b3 - eta*db3_la;

        case 'rmsprop'
          S1 = rho*S1 + (1-rho)*(dW1.^2);
          S2 = rho*S2 + (1-rho)*(dW2.^2);
          S3 = rho*S3 + (1-rho)*(dW3.^2);
          W1 = W1 - (eta./sqrt(S1+eps_opt)).*dW1; b1 = b1 - eta*db1;
          W2 = W2 - (eta./sqrt(S2+eps_opt)).*dW2; b2 = b2 - eta*db2;
          W3 = W3 - (eta./sqrt(S3+eps_opt)).*dW3; b3 = b3 - eta*db3;

        otherwise
          W1 = W1 - eta*dW1; b1 = b1 - eta*db1;
          W2 = W2 - eta*dW2; b2 = b2 - eta*db2;
          W3 = W3 - eta*dW3; b3 = b3 - eta*db3;
      end
    end % epochs

    % -----------------------------
    % Teste
    % -----------------------------
    A1t = act_forward(Xtest*W1 + b1, act1, leaky_alpha);
    A2t = act_forward(A1t*W2 + b2, act2, leaky_alpha);
    A3t = softmax_rows(A2t*W3 + b3);
    [~, pred] = max(A3t, [], 2);
    TX_OK(r) = sum(pred == Test(:,end)) / size(Test,1) * 100;
  end % repeats

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction

function A = act_forward(Z, act, alpha)
  switch act
    case 'sigmoid',    A = 1./(1+exp(-Z));
    case 'tanh',       A = tanh(Z);
    case 'relu',       A = max(Z,0);
    case 'leakyrelu',  A = max(Z,0) + alpha*min(Z,0);
    otherwise,         A = 1./(1+exp(-Z));
  end
endfunction

function D = act_backward(Z, A, act, alpha)
  switch act
    case 'sigmoid',    D = A .* (1 - A);
    case 'tanh',       D = 1 - A.^2;
    case 'relu',       D = (Z > 0);
    case 'leakyrelu',  D = (Z > 0) + alpha*(Z <= 0);
    case 'relu6',      D = (Z > 0) .* (Z < 6);
    otherwise,         D = A .* (1 - A);
  end
endfunction

function S = softmax_rows(Z)
  Zs = Z - max(Z,[],2);
  EZ = exp(Zs);
  S = EZ ./ sum(EZ,2);
endfunction
