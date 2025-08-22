clear; clc;

pkg load statistics;

D=load('recfaces.dat');
Nr = 50; Ptrain = 80;
dimensao_str = '30x30';
com_pca = 'sim';

function compara_todos_automatizado(D, Nr, Ptrain, dimensao_str, com_pca, out_csv)
  % CSV
  fid = fopen(out_csv, 'a');
  if fid == -1, error('Não foi possível abrir %s', out_csv); end
  info = dir(out_csv);
  if info.bytes == 0
    fprintf(fid, 'modelo,dimensao,numero_treinamentos,tempo_execucao_em_segundos,media,minimo,maximo,mediana,desvio_padrao,com_pca,normalizacao,funcao_ativacao,optimizador,eta,epocas\n');
  end

  % Configurações a varrer (edite aqui conforme desejar)
  normalizacoes = {'zscore','minmax11', 'minmax01', 'none'};           % reduzi por padrão; amplie se quiser
  ativacoes = {'sigmoid','tanh','relu', 'leakyrelu'};           % para MLPs
  otimizadores = {'gd','momentum','rmsprop', 'nesterov'};      % para pslog/mlp
  etas = [0.01, 0.05];
  epocas = 200;

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
  end

  fclose(fid);
endfunction


compara_todos_automatizado(D, Nr, Ptrain, dimensao_str, com_pca, 'resultados_todos_auto-v3.csv');
