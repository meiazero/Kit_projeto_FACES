function compara_todos_automatizado()
  out_csv = 'resultados_todos_sem_pca.csv'
  pkg load statistics;

  D = load('recfaces.dat');
  Nr = 50;
  Ptrain = 80;
  com_pca = 'nao';

  lines = {}

  % Ao final, escrevemos tudo de uma vez:
  fid = fopen(out_csv, 'w');
  if fid == -1, error('Não foi possível abrir %s para escrita', out_csv); end
  % Cabeçalho
  fprintf(fid, 'modelo,numero_treinamentos,tempo_execucao_em_segundos,media,minimo,maximo,mediana,desvio_padrao,r2_train,r2_test,com_pca,normalizacao,funcao_ativacao,optimizador,eta,epocas\n');
  for i=1:length(lines)
    fprintf(fid, '%s', lines{i});
  end

  % Para todos os modelos
  normalizacoes = {'zscore','minmax11', 'minmax01'};

  % Para MLPs
  ativacoes = {'sigmoid','tanh','relu', 'leakyrelu'};

  % Para pslog/mlp
  otimizadores = {'gd', 'momentum'};
  etas = [0.01];
  epocas = 200;

  for in = 1:length(normalizacoes)
    norm = normalizacoes{in};
    % ---------------- linearMQ (só normalização)
    cfg = struct('normalization', norm);
    t0 = tic;
    [STATS, TX_OK, W, R2_lin_train, R2_lin_test] = linearMQ(D, Nr, Ptrain, cfg);
    tempo = toc(t0);
    fprintf(fid, 'linearMQ,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,NaN,NaN,%d\n', ...
      Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_lin_train, R2_lin_test, com_pca, norm, epocas);
  end

  for in = 1:length(normalizacoes)
    norm = normalizacoes{in};
    % ---------------- pslog (norm x opt x eta)
    for io = 1:length(otimizadores)
      opt = otimizadores{io};
      for ie = 1:length(etas)
        eta = etas(ie);
        cfg = struct('normalization', norm, 'opt_variant', opt, 'eta', eta, 'epochs', epocas, 'mu', 0.9, 'rho', 0.9, 'eps_opt', 1e-8);
        t0 = tic;
        [STATS, TX_OK, W, R2_pslog_train, R2_pslog_test] = pslog(D, Nr, Ptrain, cfg);
        tempo = toc(t0);
        fprintf(fid, 'pslog,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,%s,%.3f,%d\n', ...
          Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_pslog_train, R2_pslog_test, com_pca, norm, opt, eta, epocas);
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
          cfg1 = struct('normalization', norm, 'hidden_act', act, 'opt_variant', opt, 'eta', eta, 'epochs', epocas);
          t0 = tic;
          [STATS, TX_OK_dummy, W1_dummy, W2_dummy, R2_mlp1h_train, R2_mlp1h_test, ] = mlp1h(D, Nr, Ptrain, cfg1);
          tempo = toc(t0);
          fprintf(fid, 'mlp1h,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s,%.3f,%d\n', ...
            Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_mlp1h_test, R2_mlp1h_train, com_pca, norm, act, opt, eta, epocas);
        end
      end
    end

    for ia = 1:length(ativacoes)
      act = ativacoes{ia};
      for io = 1:length(otimizadores)
        opt = otimizadores{io};
        for ie = 1:length(etas)
          eta = etas(ie);
          % mlp2h
          cfg2 = struct('normalization', norm, 'act1', act, 'act2', act, 'opt_variant', opt, 'eta', eta, 'epochs', epocas);
          t0 = tic;
          [STATS, TX_OK_dummy, W1_dummy, W2_dummy, W3_dummy] = mlp2h(D, Nr, Ptrain, cfg2);
          tempo = toc(t0);
          fprintf(fid, 'mlp2h,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s,%.3f,%d\n', ...
            Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_mlp1h_test, R2_mlp1h_train, com_pca, norm, act, opt, eta, epocas);

        end
      end
    end

    % ---------------- k-Nearest Neighbors (kNN)
    k_list = [1, 3, 5];
    for ik = 1:length(k_list)
      k = k_list(ik);
      cfg_knn = struct('normalization', norm, 'k', k);
      t0 = tic;
      [STATS, TX_OK, R2_knn_train, R2_knn_test] = knn(D, Nr, Ptrain, cfg_knn);
      tempo = toc(t0);
      fprintf(fid, 'knn,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,NaN,%d,0\n', ...
        Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_knn_train, R2_knn_test, com_pca, norm, k);
    end

    % ---------------- Naive Bayes (Gaussian NB)
    cfg_nb = struct('normalization', norm);
    t0 = tic;
    [STATS, TX_OK, R2_nb_train, R2_nb_test] = nb(D, Nr, Ptrain, cfg_nb);
    tempo = toc(t0);
    fprintf(fid, 'nb,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,NaN,NaN,0\n', ...
      Nr, tempo, STATS(1), STATS(2), STATS(3), STATS(4), STATS(5), R2_nb_train, R2_nb_test, com_pca, norm);

    % ---------------- Nearest Centroid (NC)
    cfg_nc = struct('normalization', norm);
    t0 = tic;
    [STATS_nc, TX_OK_nc, R2_nc_train, R2_nc_test] = nc(D, Nr, Ptrain, cfg_nc);
    tempo = toc(t0);
    fprintf(fid, 'nc,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,NaN,NaN,0\n', ...
      Nr, tempo, STATS_nc(1), STATS_nc(2), STATS_nc(3), STATS_nc(4), STATS_nc(5), R2_nc_train, R2_nc_test, com_pca, norm);

    % ---------------- Linear Discriminant Analysis (LDA)
    cfg_lda = struct('normalization', norm);
    t0 = tic;
    [STATS_lda, TX_OK_lda, R2_lda_train, R2_lda_test] = lda(D, Nr, Ptrain, cfg_lda);
    tempo = toc(t0);
    fprintf(fid, 'lda,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,NaN,NaN,NaN,0\n', ...
      Nr, tempo, STATS_lda(1), STATS_lda(2), STATS_lda(3), STATS_lda(4), STATS_lda(5), R2_lda_train, R2_lda_test, com_pca, norm);
  end

fclose(fid)
endfunction
