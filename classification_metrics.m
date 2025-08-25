function [rec, prec, f1] = classification_metrics(y_true, y_pred)
  %CLASSIFICATION_METRICS Calcula macro recall, precision e F1-score.
  %   [rec, prec, f1] = classification_metrics(y_true, y_pred)
  %   y_true: vetor de rótulos verdadeiros (1..K)
  %   y_pred: vetor de rótulos preditos (1..K)
  %   Retorna macro recall, macro precision e macro F1.
  % Determine number of classes from both true and predicted labels
  all_labels = [y_true(:); y_pred(:)];
  K = max(all_labels);
  N = length(y_true);
  % Matriz de confusão
  C = zeros(K, K);
  for i = 1:N
    if y_true(i) <= K && y_pred(i) <= K
      C(y_true(i), y_pred(i)) = C(y_true(i), y_pred(i)) + 1;
    end
  end
  recs = zeros(K,1);
  precs = zeros(K,1);
  f1s = zeros(K,1);
  for c = 1:K
    TP = C(c,c);
    FN = sum(C(c,:)) - TP;
    FP = sum(C(:,c)) - TP;
    if (TP + FN) > 0
      recs(c) = TP / (TP + FN);
    else
      recs(c) = NaN;
    end
    if (TP + FP) > 0
      precs(c) = TP / (TP + FP);
    else
      precs(c) = NaN;
    end
    if (recs(c) + precs(c)) > 0
      f1s(c) = 2 * recs(c) * precs(c) / (recs(c) + precs(c));
    else
      f1s(c) = 0;
    end
  end
  % Média ignorando classes sem ocorrência
  valid = ~isnan(recs);
  if any(valid)
    rec = mean(recs(valid));
    prec = mean(precs(valid));
  else
    rec = NaN;
    prec = NaN;
  end
  % F1 é média simples
  f1 = mean(f1s);
endfunction