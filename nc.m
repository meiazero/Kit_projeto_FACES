% Nearest Centroid classifier

function [STATS, TX_OK, R2_train_mean, R2_test_mean] = nc(D, Nr, Ptrain, config)
  if nargin < 4, config = struct(); end

  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  TX_OK = zeros(Nr,1);
  R2_train = zeros(Nr,1);
  R2_test  = zeros(Nr,1);
  epsn = 1e-8;

  % normalization variants (zscore, minmax01, minmax11, none, robust, l2, whiten)
  if ~isfield(config,'normalization'), config.normalization = 'zscore'; end

  for r = 1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test  = Dsh(Ntrain+1:end,:);
    Xtrain_raw = Train(:,1:p);
    Xtest_raw  = Test(:,1:p);

    [Xtrain, Xtest] = normalization(Xtrain_raw, Xtest_raw, config.normalization, epsn);

    Ytrain = Train(:,end);
    Ytest  = Test(:,end);

    % Centroides
    centroids = zeros(K,p);
    for c = 1:K
      Xc = Xtrain(Ytrain==c,:);
      if ~isempty(Xc)
        centroids(c,:) = mean(Xc,1);
      end
    end

    % Classificação (vetorizada)
    % d2(i,c) = ||x_i - centroid_c||^2
    % Implementação: (x^2) - 2 x*c' + (c^2)
    Xtest_sq = sum(Xtest.^2,2);
    C_sq = sum(centroids.^2,2)';
    d2 = Xtest_sq + C_sq - 2*(Xtest*centroids');
    [~, preds] = min(d2,[],2);

    M = size(Xtest,1);
    TX_OK(r) = sum(preds == Ytest)/M * 100;

    % Predição treino
    Xtrain_sq = sum(Xtrain.^2,2);
    d2_tr = Xtrain_sq + C_sq - 2*(Xtrain*centroids');
    [~, preds_train] = min(d2_tr,[],2);

    SSres_train = sum((Ytrain - preds_train).^2);
    SStot_train = sum((Ytrain - mean(Ytrain)).^2);
    R2_train(r) = 1 - SSres_train / max(SStot_train,epsn);

    SSres_test = sum((Ytest - preds).^2);
    SStot_test = sum((Ytest - mean(Ytest)).^2);
    R2_test(r) = 1 - SSres_test / max(SStot_test,epsn);
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
  R2_train_mean = mean(R2_train);
  R2_test_mean  = mean(R2_test);

  fprintf('nc: normalization: %s\n', config.normalization);
  fprintf('Stats - mean: %.3f, min: %.3f, max: %.3f, median: %.3f, std: %.3f, R2_test: %.3f, R2_train: %.3f\n', ...
    STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_test_mean, R2_train_mean);
endfunction

function [Xtrain, Xtest] = normalization(Xtr, Xte, mode, epsn)
  switch mode
    case 'zscore'
      m = mean(Xtr,1); s = std(Xtr,0,1); s(s<epsn)=1;
      Xtrain = (Xtr - m)./s; Xtest = (Xte - m)./s;
    case 'minmax01'
      mn = min(Xtr,[],1); mx = max(Xtr,[],1); rg = mx - mn; rg(rg<epsn)=1;
      Xtrain = (Xtr - mn)./rg; Xtest = (Xte - mn)./rg;
    case 'minmax11'
      mn = min(Xtr,[],1); mx = max(Xtr,[],1); rg = mx - mn; rg(rg<epsn)=1;
      Xtrain = 2*((Xtr - mn)./rg)-1; Xtest = 2*((Xte - mn)./rg)-1;
    case 'robust'
      med = median(Xtr,1);
      q1 = prctile(Xtr,25,1); q3 = prctile(Xtr,75,1);
      iqr = q3 - q1; iqr(iqr<epsn)=1;
      Xtrain = (Xtr - med)./iqr; Xtest = (Xte - med)./iqr;
    case 'maxabs'
      ma = max(abs(Xtr),[],1); ma(ma<epsn)=1;
      Xtrain = Xtr./ma; Xtest = Xte./ma;
    case 'center'
      m = mean(Xtr,1);
      Xtrain = Xtr - m; Xtest = Xte - m;
    case 'unitvar'
      s = std(Xtr,0,1); s(s<epsn)=1;
      Xtrain = Xtr./s; Xtest = Xte./s;
    case 'l2sample'
      nr = sqrt(sum(Xtr.^2,2)); nr(nr<epsn)=1;
      Xtrain = Xtr./nr;
      nr2 = sqrt(sum(Xte.^2,2)); nr2(nr2<epsn)=1;
      Xtest = Xte./nr2;
    case 'whiten'
      m = mean(Xtr,1);
      Xc = bsxfun(@minus,Xtr,m);
      C = cov(Xc,1);
      [V,S] = eig(C);
      s = diag(S); s(s<epsn)=epsn;
      W = V*diag(1./sqrt(s))*V';
      Xtrain = Xc*W;
      Xtest = bsxfun(@minus,Xte,m)*W;
    otherwise
      Xtrain = Xtr; Xtest = Xte;
  end
end