% Routines for opening face images and convert them to column vectors
% by stacking the columns of the face matrix one beneath the other.
%
% Last modification: 10/08/2021
% Author: Guilherme Barreto

clear; clc; close all;

pkg load image;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fase 1 -- Carrega imagens disponiveis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
part1 = 'subject0';
part2 = 'subject';
part3 = {'.centerlight' '.glasses' '.happy' '.leftlight' '.noglasses' '.normal' '.rightlight' '.sad' '.sleepy' '.surprised' '.wink'};
part4 = strvcat(part3);

Nind=15;   % Quantidade de individuos (classes)
Nexp=length(part3);  % Quantidade de expressoes

X=[];  % Matriz que acumula imagens vetorizadas
Y=[];  % Matriz que acumula o rotulo (identificador) do individuo
Z=[];
NAME=[];
for i=1:Nind,  % Indice para os individuos
    individuo=i,
    for j=1:Nexp,   % Indice para expressoes
        if i<10,
            nome = strcat("dataset/", part1, int2str(i), part4(j,:));    % Monta o nome do arquivo de imagem
        else
            nome = strcat("dataset/", part2, int2str(i), part4(j,:));
        end

        Img=imread(nome);  % le imagem

        Ar = imresize(Img,[20,20]);   % (Opcional) Redimensiona imagem

        An=Ar; %An=imnoise(Ar,'gaussian',0,0.005);  % (Opcional) adiciona ruido

        A=im2double(An);  % converte (im2double) para double precision

        a=A(:);  % Etapa de vetorizacao: Empilhamento das colunas

        %ROT=zeros(Nind,1); ROT(i)=1;  % Cria rotulo da imagem (binario {0,1}, one-hot encoding)
        %ROT=strcat(part1,int2str(i));
        %ROT=-ones(Nind,1); ROT(i)=1;  % Cria rotulo da imagem (bipolar {-1,+1})
        ROT = i;   % Rotulo = indice do individuo

        X=[X a]; % Coloca cada imagem vetorizada como coluna da matriz X
        Y=[Y ROT]; % Coloca o rotulo de cada vetor como coluna da matriz Y
    end
end

%%%%%%%% APLICACAO DE PCA (PCACOV) %%%%%%%%%%%
[V L VEi]=pcacov(cov(X'));
% q=39.400 é 98% de explicados, escolhido por mim
% q=35.294 é 98% segundo as variavel VEi
q=35.294; Vq=V(:,1:q); Qq=Vq'; X=Qq*X;
VEq=cumsum(VEi);

figure; plot(VEq,'r-','linewidth',3); xlabel('Autovalor'); ylabel('Variancia explicada acumulada');

Z=[X;Y];  % Formato 01 vetor de atributos por coluna: DIM(Z) = (p+1)xN
Z=Z';     % Formato 01 vetor de atributos por linha: DIM(Z) = Nx(p+1)

save -ascii recfaces.dat Z

%save -ascii yale1_input20x20.txt X
%save -ascii yale1_output20x20.txt Y



