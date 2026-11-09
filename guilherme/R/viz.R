library(arrow)
library(readr)


df <- read_parquet("/home/guilhermegarber/trabalho-cm263/guilherme/dados/gercon_poa_censo.parquet", as_data_frame = FALSE)

gercon_poa_censo <- head(as.data.frame(df), 1000)

View(gercon_poa_censo)


df <- read_parquet("/home/guilhermegarber/trabalho-cm263/guilherme/dados/gercon_poa_censo_filtrado.parquet", as_data_frame = FALSE)

gercon_poa_censo_filtrado <- head(as.data.frame(df), 1000)

View(gercon_poa_censo_filtrado)


df <- read_parquet("/home/guilhermegarber/trabalho-cm263/guilherme/dados/base_distancias_onibus.parquet", as_data_frame = FALSE)

base_distancias_onibus <- head(as.data.frame(df), 1000)

View(base_distancias_onibus)


df <- read_csv("/home/guilhermegarber/trabalho-cm263/guilherme/dados/dados_guilherme.csv")

dados_guilherme <- head(as.data.frame(df), 1000)

View(dados_guilherme)


df <- read_parquet("/home/guilhermegarber/trabalho-cm263/guilherme/dados/gercon_poa_pronto.parquet")

gercon_poa_pronto <- head(as.data.frame(df), 1000)

View(gercon_poa_pronto)

