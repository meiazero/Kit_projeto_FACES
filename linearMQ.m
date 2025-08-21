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

    % Normalize features (z-score)
    Xtrain_norm = (Xtrain_raw - m) ./ s;
    Xtrain = [ones(Ntrain,1) Xtrain_norm];

    % Normalize features (mudanca de escala [0,+1])
    % Xtrain = (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));
    % Xtest = (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));

    % Normalize features (mudanca de escala [-1,+1])
    % Xtrain = 2 * (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;
    % Xtest = 2 * (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;

    % One-hot encode labels
    Ytrain = zeros(Ntrain, K);
    for i=1:Ntrain
      Ytrain(i, Train(i,end)) = 1;
    end

    % Compute weights using pseudoinverse
    W = pinv(Xtrain) * Ytrain;
    Xtest_raw = Test(:,1:p);
    Xtest_norm = (Xtest_raw - m) ./ s;
    Xtest = [ones(size(Test,1),1) Xtest_norm];
    Ypred = Xtest * W;

    [~, pred] = max(Ypred, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;

    % Log progress
    current_mean = mean(TX_OK(1:r));
    current_std = std(TX_OK(1:r));
    % disp(["<MQ, round = " num2str(r) "/" num2str(Nr) ", mean acc = " num2str(current_mean) ", std acc = " num2str(current_std) ">"]);
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];

  % Plot accuracy over rounds
  % figure;
  % plot(1:Nr, TX_OK, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
  % title('Accuracy over Rounds for MQ', 'FontSize', 14, 'FontWeight', 'bold');
  % xlabel('Round Number', 'FontSize', 12);
  % ylabel('Test Accuracy (%)', 'FontSize', 12);
  % grid on;
  % set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
  % print -dpng 'mq_accuracy_plot.png';

endfunction