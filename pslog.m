function [STATS TX_OK W] = pslog(D, Nr, Ptrain)
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
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
    % Z-score (recomendado)
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
    % Inicialização W (pesos)
    % -----------------------------
    % W = randn(p+1, K) * 0.1;           % alternativa simples
    W = randn(p+1, K) * sqrt(2/(p+K));  % inicialização razoável (similar Glorot)

    Xtrain_b = [ones(Ntrain,1) Xtrain];

    % -----------------------------
    % Hiperparâmetros (descomente/ajuste)
    % -----------------------------
    eta = 0.01;      % taxa de aprendizado
    epochs = 200;    % épocas
    % variantes de otimizador: 'gd' (padrão), 'momentum', 'nesterov', 'rmsprop'
    opt_variant = 'gd';

    % Parâmetros de momentum / RMSProp (se usar)
    mu = 0.9;
    rho = 0.9;
    eps_opt = 1e-8;

    % Estados para otimizadores (usados se descomentados)
    V = zeros(size(W));  % momentum/nesterov
    S = zeros(size(W));  % rmsprop cache

    % -----------------------------
    % Loop de treinamento (batch GD)
    % -----------------------------
    for e=1:epochs
      Z = Xtrain_b * W;          % (Ntrain x K)
      P = softmax_rows(Z);       % probabilidades (Ntrain x K)

      % Gradiente (média por amostra)
      G = (Xtrain_b' * (P - Ytrain_oh)) / Ntrain;

      switch opt_variant
        case 'gd'
          W = W - eta * G;

        case 'momentum'
          V = mu * V + eta * G;
          W = W - V;

        case 'nesterov'
          % Nesterov (aprox)
          W_look = W - mu * V;
          P_la = softmax_rows(Xtrain_b * W_look);
          G_la = (Xtrain_b' * (P_la - Ytrain_oh)) / Ntrain;
          V = mu * V + eta * G_la;
          W = W - V;

        case 'rmsprop'
          S = rho * S + (1 - rho) * (G.^2);
          W = W - (eta ./ sqrt(S + eps_opt)) .* G;

        otherwise
          W = W - eta * G;
      end
    end

    % -----------------------------
    % Teste
    % -----------------------------
    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ztest = Xtest_b * W;
    [~, pred] = max(Ztest, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
endfunction

function S = softmax_rows(Z)
  Zs = Z - max(Z,[],2);
  EZ = exp(Zs);
  S = EZ ./ sum(EZ,2);
endfunction
