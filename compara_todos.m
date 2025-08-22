clear; clc;

pkg load statistics;

D=load('recfaces.dat');

Nr=50;  % No. de repeticoes

Ptrain=80; % Porcentagem de treinamento

tic; [linearMQModel TX_OK5 W]=linearMQ(D,Nr,Ptrain); TempoMQ=toc; % Classificador Linear de Minimos Quadrados
tic; [pslogModel TX_OK6 W]=pslog(D,Nr,Ptrain); TempoPSlog=toc; % Perceptron Logistico
tic; [mpl1hModel TX_OK7 W]=mlp1h(D,Nr,Ptrain); TempoMLP1h=toc; % MLP 1-hidden
tic; [mlp2hModel TX_OK8 W]=mlp2h(D,Nr,Ptrain); TempoMLP2h=toc; % MLP 2-hidden

disp("\nTempo de Execução, Média, Minimo, maximo, mediana, desvio padrão")
disp(["linearMQ: " num2str(TempoMQ) "," num2str(linearMQModel(1)) "," num2str(linearMQModel(2)) "," num2str(linearMQModel(3)) "," num2str(linearMQModel(4)) "," num2str(linearMQModel(5))])
disp(["PSLog: " num2str(TempoPSlog) "," num2str(pslogModel(1)) "," num2str(pslogModel(2)) "," num2str(pslogModel(3)) "," num2str(pslogModel(4)) "," num2str(pslogModel(5))])
disp(["MLP1h: " num2str(TempoMLP1h) "," num2str(mpl1hModel(1)) "," num2str(mpl1hModel(2)) "," num2str(mpl1hModel(3)) "," num2str(mpl1hModel(4)) "," num2str(mpl1hModel(5))])
disp(["MLP2h: " num2str(TempoMLP2h) "," num2str(mlp2hModel(1)) "," num2str(mlp2hModel(2)) "," num2str(mlp2hModel(3)) "," num2str(mlp2hModel(4)) "," num2str(mlp2hModel(5))])

% disp([num2str(TempoMQ) "\n" num2str(TempoPSlog) "\n" num2str(TempoMLP1h) "\n" num2str(TempoMLP2h)])

% boxplot([TX_OK0' TX_OK1' TX_OK2' TX_OK3' TX_OK4' TX_OK5'])
% set(gca (), "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"Quadratico","Variante 1", "Variante 2","Variante 3","Variante 4","MQ"})
% boxplot([TX_OK5' TX_OK6' TX_OK7' TX_OK8'])
% set(gca, "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"MQ", "PSLog", "MLP1", "MLP2"})
% set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
% grid on;
% title('Conjunto Coluna');
% xlabel('Classificador');
% ylabel('Taxas de acerto');
