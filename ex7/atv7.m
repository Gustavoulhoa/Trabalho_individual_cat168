vazio = [];

% Script adaptado para Previsão de Carga Elétrica com Rede Neural Temporal
% Baseado no modelo estruturado pelo aplicativo Neural Time Series

% 1. CARREGAR OS SEUS DADOS REAIS
% Garante que a rede usará os seus dados de potência ativa importados
X = dados_series; % Série temporal (936 passos)
T = dados_series; % Como é autorregressiva, o alvo é a própria série

% 2. CONFIGURAÇÃO DA REDE (Conforme o enunciado)
trainFcn = 'trainlm';       % Levenberg-Marquardt
inputDelays = 1:24;         % Atrasos de tempo de 1 a 24
hiddenLayerSize = 10;       % 10 neurônios na camada oculta

% Criar a Rede Temporal de Atraso
net = timedelaynet(inputDelays, hiddenLayerSize, trainFcn);

% Separar as últimas 24 horas para o Teste Cego antes do treino
n_teste = 24;
X_treino = X(1:end-n_teste);
T_treino = T(1:end-n_teste);

% 3. PREPARAÇÃO E TREINAMENTO (Malha Aberta)
[x, xi, ai, t] = preparets(net, X_treino, T_treino);

% Divisão dos dados de treino (70% treino, 15% validação, 15% teste interno)
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Treinar a rede
fprintf('Treinando a rede neural com os dados históricos...\n');
[net, tr] = train(net, x, t, xi, ai);

% 4. PREVISÃO RECURSIVA (Fechar a Malha para os 24 passos do Teste Cego)
fprintf('Fechando a malha para realizar a previsão recursiva de 24 horas...\n');

% Converte a rede treinada para Malha Fechada (Closed Loop)
net_fechada = closeloop(net);

% Para fazer a previsão recursiva, precisamos fornecer os últimos 24 dados reais do treino
% como condições iniciais (estados de atraso) para a rede começar a prever sozinha
X_teste_total = X(end-n_teste-max(inputDelays)+1:end);
T_teste_total = T(end-n_teste-max(inputDelays)+1:end);

[x_fechada, xi_fechada, ai_fechada, t_fechada] = preparets(net_fechada, X_teste_total, T_teste_total);

% Executa a rede em malha fechada para obter os 24 passos previstos recursivamente
y_recursivo = net_fechada(x_fechada, xi_fechada, ai_fechada);

% 5. CONVERTER RESULTADOS PARA O CALCULO DAS MÉTRICAS
% Transforma os Cell Arrays de volta em vetores numéricos comuns para as fórmulas
pot_medida = cell2mat(t_fechada)';
pot_prevista = cell2mat(y_recursivo)';

% Assegurar que os vetores são do tamanho exato do teste cego (24 passos)
pot_medida = pot_medida(1:n_teste);
pot_prevista = pot_prevista(1:n_teste);

% 6. CÁLCULO DAS MÉTRICAS DE DESEMPENHO
% Fórmulas solicitadas no enunciado do exercício
mape = mean(abs((pot_medida - pot_prevista) ./ pot_medida)) * 100;
rmse = sqrt(mean((pot_medida - pot_prevista).^2));

% 7. EXIBIÇÃO DOS RESULTADOS E GRÁFICO
fprintf('\n================ RESULTADOS FINAIS ================＼n');
fprintf('MAPE (Erro Médio Absoluto Percentual): %.2f %%\n', mape);
fprintf('RMSE (Raiz do Erro Quadrático Médio): %.2f MW\n', rmse);
fprintf('==================================================＼n');

% Plotar gráfico comparativo para o seu relatório
figure;
plot(pot_medida, 'b-o', 'LineWidth', 1.5); hold on;
plot(pot_prevista, 'r--x', 'LineWidth', 1.5);
legend('Real (Medido)', 'Previsto (Recursivo de 1 a 24h)');
title('Teste Cego - Previsão Recursiva de Carga Elétrica');
xlabel('Passos à Frente (Horas)'); ylabel('Potência Ativa (MW)');
grid on;
