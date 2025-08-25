function compara_todos_automatizado()
  % COMPARA_TODOS_AUTOMATIZADO Executa vários classificadores e salva em CSV.
  % out_csv: nome do arquivo CSV de saída (opcional).
  out_csv = 'resultados_todos_sem_pca.csv'
  pkg load statistics;  % necessário para normalização e estatísticas

  % Carrega dados e parâmetros padrão
  D = load('recfaces.dat');
  Nr = 50;
  Ptrain = 80;
  dimensao_str = '30x30';
  com_pca = 'nao';

  % Abre arquivo CSV para anexar resultados
  [fid, errmsg] = fopen(out_csv, 'a');
  if fid == -1
    error('Não foi possível abrir %s: %s', out_csv, errmsg);
  end
  info = dir(out_csv);
  if info.bytes == 0
    header = ['modelo,dimensao,numero_treinamentos,tempo_execucao_em_segundos,' ...
              'media,minimo,maximo,mediana,desvio_padrao,com_pca,normalizacao,' ...
              'funcao_ativacao,optimizador,eta,epocas\n'];
    fprintf(fid, '%s', header);
  end

  normalizacoes = {'zscore','minmax11', 'minmax01', 'none'};           % reduzi por padrão; amplie se quiser
  ativacoes = {'sigmoid','tanh','relu', 'leakyrelu'};           % para MLPs
  otimizadores = {'gd','momentum','rmsprop', 'nesterov'};      % para pslog/mlp
  etas = [0.001, 0.003, 0.005];
  epocas = 100;

  for in = 1:length(normalizacoes)
    norm = normalizacoes{in};

    % ---------------- linearMQ (só normalização)
    cfg = struct('normalization', norm);
    t0 = tic;
    [STATS, TX_OK, W] = linearMQ(D, Nr, Ptrain, cfg);
    tempo = toc(t0);
    fprintf(fid, 'linearMQ,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s,%s,%.3f,%d\n', ...
      dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
      com_pca, norm, 'na', 'na', NaN, epocas);

    % ---------------- pslog (norm x opt x eta)
    for io = 1:length(otimizadores)
      opt = otimizadores{io};
      for ie = 1:length(etas)
        eta = etas(ie);
        cfg = struct('normalization', norm, 'opt_variant', opt, 'eta', eta, 'epochs', epocas, 'mu', 0.9, 'rho', 0.9, 'eps_opt', 1e-8);
        t0 = tic;
        [STATS, TX_OK, W] = pslog(D, Nr, Ptrain, cfg);
        tempo = toc(t0);
        fprintf(fid, 'pslog,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s,%s,%.3f,%d\n', ...
          dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
          com_pca, norm, 'na', opt, eta, epocas);
      end
    end

    % ---------------- MLPs (norm x act x opt x eta)
    for ia = 1:length(ativacoes)
      act = ativacoes{ia};
      for io = 1:length(otimizadores)
        opt = otimizadores{io};
        for ie = 1:length(etas)
          eta = etas(ie);

          % mlp1h
          cfg1 = struct('normalization', norm, 'hidden_act', act, 'opt_variant', opt, 'eta', eta, 'epochs', epocas, 'mu', 0.9, 'rho', 0.9, 'eps_opt', 1e-8, 'leaky_alpha', 0.01);
          t0 = tic;
          [STATS, TX_OK, W1, W2] = mlp1h(D, Nr, Ptrain, cfg1);
          tempo = toc(t0);
          fprintf(fid, 'mlp1h,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s,%s,%.3f,%d\n', ...
            dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
            com_pca, norm, act, opt, eta, epocas);

          % mlp2h
          cfg2 = struct('normalization', norm, 'act1', act, 'act2', act, 'opt_variant', opt, 'eta', eta, 'epochs', epocas, 'mu', 0.9, 'rho', 0.9, 'eps_opt', 1e-8, 'leaky_alpha', 0.01);
          t0 = tic;
          [STATS, TX_OK, W1, W2, W3] = mlp2h(D, Nr, Ptrain, cfg2);
          tempo = toc(t0);
          fprintf(fid, 'mlp2h,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s,%s,%.3f,%d\n', ...
            dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
            com_pca, norm, act, opt, eta, epocas);
        end
      end
    end

    % ---------------- k-Nearest Neighbors (kNN)
    k_list = [1, 3, 5];
    for ik = 1:length(k_list)
      k = k_list(ik);
      cfg_knn = struct('normalization', norm, 'k', k);
      t0 = tic;
      [STATS, TX_OK] = knn(D, Nr, Ptrain, cfg_knn);
      tempo = toc(t0);
      fprintf(fid, 'knn,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,na,na,%d,0\n', ...
        dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
        com_pca, norm, k);
    end

    % ---------------- Naive Bayes (Gaussian NB)
    cfg_nb = struct('normalization', norm);
    t0 = tic;
    [STATS, TX_OK] = nb(D, Nr, Ptrain, cfg_nb);
    tempo = toc(t0);
    fprintf(fid, 'nb,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,na,na,NaN,0\n', ...
      dimensao_str, Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), ...
      com_pca, norm);

    % ---------------- Nearest Centroid (NC)
    cfg_nc = struct('normalization', norm);
    t0 = tic;
    [STATS_nc, TX_OK_nc] = nc(D, Nr, Ptrain, cfg_nc);
    tempo = toc(t0);
    fprintf(fid, 'nc,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,na,na,NaN,0\n', ...
      dimensao_str, Nr, tempo, STATS_nc(1), STATS_nc(2), STATS_nc(3), STATS_nc(4), STATS_nc(5), ...
      com_pca, norm);

    % ---------------- Linear Discriminant Analysis (LDA)
    cfg_lda = struct('normalization', norm);
    t0 = tic;
    [STATS_lda, TX_OK_lda] = lda(D, Nr, Ptrain, cfg_lda);
    tempo = toc(t0);
    fprintf(fid, 'lda,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,na,na,NaN,0\n', ...
      dimensao_str, Nr, tempo, STATS_lda(1), STATS_lda(2), STATS_lda(3), STATS_lda(4), STATS_lda(5), ...
      com_pca, norm);
  end

  fclose(fid);
endfunction
