# Projeto Aplicado III

**HistFlix: um sistema de recomendação personalizado de filmes históricos e documentários**

## 📱 Grupo:
* BRUNO BALTUILHE - 10424822 - ✉️ 10424822@mackenzista.com.br 
* ISAQUE PIMENTEL – 10415608 – ✉️ 10415608@mackenzista.com.br
* KELLY GRAZIELY PENA - 10416108 - ✉️ 10416108@mackenzista.com.br
  
## 💡 Apresentação do Grupo

Somos um grupo de alunos de Ciências de Dados desenvolvendo um projeto de Sistema de Recomendação para melhorar as técnicas aprendizagem de disciplinas escolares além da sala de aula.

Propomos o **HistFlix**, um sistema de recomendação de filmes e documentários de qualidade e relevância histórica e educacional, para estender o aprendizado da História além da aula de aula. 
Após a desenvolvimento do nosso produto, o apresentaremos para avaliação da disciplina de Projeto Aplicado III da Universidade Mackenzie.

## 🔍 Objetivo do Projeto

🎯 **Objetivo Geral**

Desenvolver um sistema de recomendação de filmes, séries e documentários que utilize um modelo híbrido de recomendação e inteligência artificial para interpretar as emoções do usuário e sugerir conteúdos alinhados ao seu estado emocional e preferências específicas, com o propósito de aumentar o interesse pela história e democratizar o acesso a conteúdos audiovisuais educativos. 

🛠️ **Objetivos Específicos**
- Coletar e processar dados sobre filmes e documentários, utilizando a base de dados MovieLens, para criar um modelo de recomendação personalizado.
- Implementar um modelo híbrido de recomendação, combinando filtragem colaborativa (que analisa o comportamento e as avaliações de outros usuários com perfis semelhantes) e filtragem baseada em conteúdo (que considera características específicas das obras audiovisuais, como gênero, duração, elenco e temática). 
- Desenvolver uma interface interativa na qual os usuários possam expressar suas emoções e preferências momentâneas.
- Integrar técnicas de Processamento de Linguagem Natural (PLN) para interpretar sentimentos e preferências expressas textualmente.

## ✨ Motivação e justificativas

A escolha do tema é impulsionada pelo crescente interesse em métodos de ensino alternativos que possam complementar os modelos tradicionais de educação. Filmes, séries e documentários despertam interesse em diferentes temas, como cultura, história e ciência, através de uma abordagem lúdica e visual. Além disso, o avanço da inteligência artificial permite que sistemas de recomendação personalizem essas experiências, sugerindo conteúdos alinhados às preferências dos usuários e suas necessidades de aprendizado.

A HistFlix busca suprir a necessidade de um sistema especializado que forneça recomendações precisas e relevantes para estudantes, pesquisadores e entusiastas da história. Além disso, o projeto alinha-se aos Objetivos de Desenvolvimento Sustentável (ODS) da ONU, promovendo educação de qualidade ao facilitar o acesso a conteúdos educativos.

## 📅 Cronograma de Desenvolvimento  

| **Etapa**  | **Atividade**  | **Prazo**  | **Impacto Expansionista**  |  
|------------|----------------|------------|---------------------------------|  
| **Etapa 1**  | Concepção do Produto  | Semana 1-2  | Identificação de necessidades da comunidade e levantamento de filmes relevantes para educação histórica.  |  
| **Etapa 2**  | Definição do Produto  | Semana 3-4  | Validar critérios de recomendação e ampliar a abrangência do sistema.  |  
| **Etapa 3**  | Metodologia e Implementação do Modelo e Testes  | Semana 5-6  | Desenvolvimento do sistema com feedback de usuários.  |  
| **Etapa 4**  | Resultado e Conclusão  | Semana 7-8  | Apresentação do projeto.  |  


## Bibliotecas Python

Abaixo está uma lista provável de bibliotecas Python que serão utilizadas no desenvolvimento do projeto:

- **pandas**: Para manipulação e análise de dados estruturados.
- **numpy**: Para operações matemáticas e manipulação de arrays numéricos.
- **sqlite3**: Para interação com o banco de dados SQLite.
- **scikit-learn**: Para implementação de algoritmos de aprendizado de máquina e métricas de avaliação.
- **surprise**: Para construção e avaliação de sistemas de recomendação.
- **matplotlib**: Para criação de gráficos e visualizações.
- **seaborn**: Para visualizações estatísticas mais avançadas e estilizadas.
- **pytest**: Para criação e execução de testes automatizados.
- **os**: Para manipulação de caminhos e arquivos no sistema operacional.
- **logging**: Para registro de logs e monitoramento do sistema.

## Etapa de Extração dos Dados

A etapa de extração dos dados consiste em transformar os arquivos originais do MovieLens 1M, que estão no formato `.dat`, em tabelas estruturadas dentro de um banco de dados relacional SQLite. Essa transformação é essencial para facilitar a manipulação, consulta e análise dos dados durante o desenvolvimento do sistema de recomendação.

### Passos Realizados:

1. **Leitura dos Arquivos `.dat`**:
   - Os arquivos `users.dat`, `ratings.dat` e `movies.dat` são lidos utilizando a biblioteca `pandas`.
   - Cada arquivo é carregado em um DataFrame, com as colunas devidamente nomeadas de acordo com a documentação do MovieLens 1M.

2. **Limpeza e Normalização dos Dados**:
   - Remoção de valores ausentes ou inconsistentes.
   - Conversão de tipos de dados para formatos mais eficientes (e.g., int32, float32).
   - Normalização de colunas, como transformar o gênero em valores numéricos (e.g., 0 para "F" e 1 para "M").
   - Extração de informações adicionais, como o ano de lançamento dos filmes a partir do título.

3. **Criação do Banco de Dados SQLite**:
   - Um banco de dados SQLite é criado utilizando a biblioteca `sqlite3`.
   - As tabelas `users`, `ratings` e `movies` são criadas no banco de dados, e os dados limpos são inseridos diretamente a partir dos DataFrames.

4. **Armazenamento Estruturado**:
   - Os dados são armazenados no banco de dados SQLite, permitindo consultas SQL eficientes e integração com outras ferramentas de análise.

### Benefícios:
- **Eficiência**: A estrutura relacional do SQLite permite consultas rápidas e organizadas.
- **Portabilidade**: O banco de dados SQLite é leve e pode ser facilmente compartilhado ou integrado ao sistema.
- **Facilidade de Manipulação**: A utilização de SQL simplifica a extração de informações específicas para análises ou treinamento de modelos.

## Análise Exploratória (EDA)

A Análise Exploratória de Dados (EDA) foi conduzida para entender melhor o comportamento dos usuários, padrões de avaliação e a qualidade da base de dados. Os principais pontos analisados foram:

Distribuição das Avaliações: Identificação da tendência dos usuários em dar notas mais altas e os padrões gerais de avaliação.

![Distribuição das Avaliações](pictures/rating_distribution.png)

Perfil dos Usuários: Análise de distribuição por gênero e faixa etária.

![Perfil dos Usuários - Gênero](pictures/gender_distribution.png)
![Perfil dos Usuários - Idade](pictures/age_distribution.png)

Perfil dos Filmes: Identificação dos gßênero melhores avaliados por gênero.

![Perfil dos Filmes](pictures/average_rating_by_genre_gender.png)

Evolução das Avaliações: Observação do comportamento das avaliações ao longo do tempo.

![Evolução das Avaliações](pictures/rating_evolution.png)

A EDA impacta diretamente o sistema de recomendação, pois permite ajustar o modelo para melhor atender aos perfis de usuários e identificar padrões de consumo de conteúdo. Para a versão em produção, consideramos utilizar uma base de dados maior, como o MovieLens 10M, para aprimorar a qualidade das recomendações e garantir maior robustez ao modelo.

## Avaliação de Desempenho

A avaliação do desempenho do modelo de recomendação será realizada utilizando as seguintes métricas:

RMSE (Root Mean Squared Error): mede a precisão da previsão das notas, comparando-as com os valores reais fornecidos pelos usuários.

Precisão@K e Recall@K: avaliam a relevância das recomendações dentro do top-K recomendações feitas para cada usuário.

Cobertura: mede a proporção do catálogo que está sendo recomendado aos usuários, garantindo diversidade.

Diversidade: mede a variação entre os itens recomendados para um mesmo usuário.


## 📚 Referencial Teórico

Este referencial teórico fundamenta as escolhas metodológicas e técnicas adotadas no desenvolvimento do **HistFlix**, nosso sistema de recomendação inteligente voltado à sugestão personalizada de filmes e documentários históricos, sensível ao estado emocional dos usuários. Para isso, são abordadas teorias, modelos e algoritmos amplamente utilizados em sistemas de recomendação, com base em estudos consolidados e pesquisas recentes.

Além disso, esta seção contextualiza a aplicação de técnicas de Processamento de Linguagem Natural (NLP) na interpretação subjetiva das interações dos usuários e discute a sinergia entre essas abordagens. Ao reunir conceitos de filtragem colaborativa, filtragem baseada em conteúdo, modelos híbridos e análise de emoções, o referencial teórico embasa a proposta de um sistema robusto, contextual e orientado ao engajamento educacional.

## 🌐 Web Application (HistFlix)

### Descrição Geral

A aplicação web HistFlix oferece uma interface amigável e bilíngue (português/inglês) para recomendações personalizadas de filmes históricos e documentários. O sistema utiliza modelos híbridos de recomendação e análise de sentimentos para sugerir conteúdos alinhados ao perfil e ao estado emocional do usuário.

#### Principais Páginas
- **Página Inicial:** Apresenta o sistema e permite ao usuário escolher entre recomendações híbridas ou baseadas em sentimento.  
   ![Página Inicial](pictures/home_page.png)

- **Recomendação Híbrida:** Permite ao usuário obter recomendações personalizadas combinando filtragem colaborativa e baseada em conteúdo. O usuário pode informar preferências e receber sugestões detalhadas.  
   ![Recomendação Híbrida](pictures/hybrid_recommendation.png)

- **Recomendação por Sentimento:** O usuário pode expressar seu estado emocional (texto ou seleção de emoção), e o sistema sugere filmes/documentários alinhados ao sentimento identificado.  
   ![Recomendação por Sentimento](pictures/sentiment_recommendation.png)
- **Página de Erro:** Exibe mensagens de erro amigáveis e bilíngues em caso de problemas ou entradas inválidas.

A navegação é simples, com suporte a troca de idioma e mensagens claras em todas as páginas.

### Como Executar a Aplicação Web

Você pode rodar a aplicação de três formas principais a partir do diretório raiz do projeto:

#### 1. Usando o script para Windows (CMD)
```bash
./run_web_app.cmd
```

#### 2. Usando o script para Unix/Linux/Mac (SH)
```bash
./run_web_app.sh
```

#### 3. Executando diretamente o arquivo Python
```bash
python -m web.web_app
```
Ou, alternativamente:
```bash
python web/web_app.py
```

A aplicação estará disponível em `http://127.0.0.1:5000/` por padrão.

### Requisitos
- Python 3.11+
- Instale as dependências com:
```bash
pip install -r requirements.txt
```

### Observações Importantes
- O sistema utiliza arquivos de modelo e banco de dados localizados nas pastas `models/` e `dataset/`.
- Para funcionamento completo, certifique-se de que os arquivos `.pkl` e o banco SQLite estejam presentes.
- O idioma pode ser alternado nas páginas da aplicação.
- Logs de execução e erros são salvos na pasta `logs/`.
