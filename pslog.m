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

    % Normalize features (z-score)
    Xtest_raw = Test(:,1:p);
    Xtrain = (Xtrain_raw - m) ./ s;
    Xtest = (Xtest_raw - m) ./ s;

    % Normalize features (mudanca de escala [0,+1])
    % Xtrain = (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));
    % Xtest = (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));

    % Normalize features (mudanca de escala [-1,+1])
    % Xtrain = 2 * (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;
    % Xtest = 2 * (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;

    % One-hot encode labels
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain
      Ytrain_oh(i, Train(i,end)) = 1;
    end

    % Initialize weights
    W = rand(p+1, K) * 0.1;
    Xtrain_b = [ones(Ntrain,1) Xtrain];
    eta = 0.01; epochs = 200;
    losses = zeros(epochs,1);

    % Training loop
    for e=1:epochs
      Z = Xtrain_b * W;
      P = softmax(Z')';

      % Compute cross-entropy loss
      loss = -sum(sum(Ytrain_oh .* log(P + 1e-10))) / Ntrain;
      losses(e) = loss;
      grad = Xtrain_b' * (P - Ytrain_oh);
      W = W - eta * grad;

      % Log every 10 epochs
      if mod(e, 10) == 0
        % disp(["<PSLog, round = " num2str(r) ", epoch = " num2str(e) "/" num2str(epochs) ", loss = " num2str(loss) ">"]);
      end
    end

    Xtest_b = [ones(size(Xtest,1),1) Xtest];
    Ztest = Xtest_b * W;
    [~, pred] = max(Ztest, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;

    % Log progress
    current_mean = mean(TX_OK(1:r));
    current_std = std(TX_OK(1:r));
    % disp(["<PSLog, round = " num2str(r) "/" num2str(Nr) ", mean acc = " num2str(current_mean) ", std acc = " num2str(current_std) ", final loss = " num2str(losses(end)) ">"]);

    % % Plot loss for this round if last
    % if r == Nr
    %   figure;
    %   plot(1:epochs, losses, 'g-', 'LineWidth', 2);
    %   title('Training Loss over Epochs for PSLog (Last Round)', 'FontSize', 14, 'FontWeight', 'bold');
    %   xlabel('Epoch', 'FontSize', 12);
    %   ylabel('Cross-Entropy Loss', 'FontSize', 12);
    %   grid on;
    %   set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
    %   print -dpng 'pslog_loss_plot.png';
    % end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];

  % Plot accuracy over rounds
  % figure;
  % plot(1:Nr, TX_OK, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
  % title('Accuracy over Rounds for PSLog', 'FontSize', 14, 'FontWeight', 'bold');
  % xlabel('Round Number', 'FontSize', 12);
  % ylabel('Test Accuracy (%)', 'FontSize', 12);
  % grid on;
  % set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
  % print -dpng 'pslog_accuracy_plot.png';
endfunction

function s = softmax(x)
  ex = exp(x - max(x,[],2));
  s = ex ./ sum(ex,2);
endfunction