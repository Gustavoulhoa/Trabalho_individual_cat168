clc
clear
close all

dados = readtable('dadosTrafos.xlsx');

X = table2array(dados(:,1:3));
T = table2array(dados(:,4));

P = X';
T = T';

size(P)
size(T)

figure
plot3(P(1,:),P(2,:),P(3,:),'o')
grid on
xlabel('x1')
ylabel('x2')
zlabel('x3')

which feedforwardnet

rng(1)

net1 = feedforwardnet(5);

net1.layers{1}.transferFcn = 'tansig';
net1.layers{2}.transferFcn = 'tansig';

[net1,tr1] = train(net1,P,T);

rng(100)

net2 = feedforwardnet(5);

net2.layers{1}.transferFcn = 'tansig';
net2.layers{2}.transferFcn = 'tansig';

[net2,tr2] = train(net2,P,T);

% Dados de teste da questão

Xteste = [
    -0.3565 -0.7842  0.3012  0.7757  0.1570 -0.7014  0.3748 -0.6920 -1.3970 -1.8842;
    0.0620  1.1267  0.5611  1.0648  0.8028  1.0316  0.1536  0.9404  0.7141 -0.2805;
    5.9891  5.5912  5.8234  8.0677  6.3040  3.6005  6.1537  4.4058  4.9263  1.2548];

% Rede 1

Y1 = net1(Xteste);
classe1 = sign(Y1);

% Rede 2

Y2 = net2(Xteste);
classe2 = sign(Y2);

disp('Saidas Rede 1')
disp(Y1)

disp('Classes Rede 1')
disp(classe1)

disp('Saidas Rede 2')
disp(Y2)

disp('Classes Rede 2')
disp(classe2)