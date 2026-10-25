# quero carregar o arquivo gercon_poa_censo_filtrado.parquet
library(arrow)
library(dplyr)
library(stringr)

gercon_poa_censo_filtrado <- read_parquet("dados/processados/gercon_poa_censo_filtrado.parquet")

glimpse(gercon_poa_censo_filtrado)

# padronizar os dados de cnes_solicitante e cnesexecutante para que fiquem no mesmo formato,
# ou seja, com 7 digitos, completando com zeros a esquerda quando necessário.
gercon_poa_censo_filtrado <- gercon_poa_censo_filtrado %>%
  mutate(
    cnes_solicitante = str_pad(cnes_solicitante, width = 7, side = "left", pad = "0"),
    cnesexecutante = str_pad(cnesexecutante, width = 7, side = "left", pad = "0")
  )

# verificar as combinações de cnes_solicitante e cnesexecutante
unique(gercon_poa_censo_filtrado[, c("cnes_solicitante", "cnesexecutante")])

# 9,148 combinações únicas de cnes_solicitante e cnesexecutante

#### Endereços CNES ####
# carregar os endereços do arquivo tbEstabelecimento202609.csv
est <- read.csv2(
  "tbEstabelecimento202609.csv",
  sep = ";",
  fileEncoding = "ISO-8859-1",
  colClasses = "character",
  check.names = FALSE
)

## CNES unicos da APS
CNES_APS<- unique(gercon_poa_censo_filtrado$cnes_solicitante)

## CNES unicos Atenção especializada 
CNES_AE<- unique(gercon_poa_censo_filtrado$cnesexecutante)

## Criar dataframe CNES APS
df_APS <- est %>%
  filter(CO_CNES %in% CNES_APS) %>%
  select(CO_CNES, NO_FANTASIA,NO_LOGRADOURO,NU_ENDERECO,NO_COMPLEMENTO,NO_BAIRRO,CO_CEP,
    NU_LATITUDE,NU_LONGITUDE)

df_AE <- est %>%
  filter(CO_CNES %in% CNES_AE) %>%
  select(CO_CNES, NO_FANTASIA,NO_LOGRADOURO,NU_ENDERECO,NO_COMPLEMENTO,NO_BAIRRO,CO_CEP,
    NU_LATITUDE,NU_LONGITUDE)

# Quantos NO_LOGRADOURO e NU_ENDERECO são iguais em df_APS, ou seja, ficam no mesmo endereço?
df_APS %>%
  group_by(NO_LOGRADOURO, NU_ENDERECO) %>%
  summarise(n = n()) %>%
  filter(n > 1) %>%
  arrange(desc(n))

# Quantos NO_LOGRADOURO e NU_ENDERECO são iguais em df_AE, ou seja, ficam no mesmo endereço?
df_AE %>%
  group_by(NO_LOGRADOURO, NU_ENDERECO) %>%
  summarise(n = n()) %>%
  filter(n > 1) %>%
  arrange(desc(n))

# Merge df_APS e df_AE  
df_merged <- full_join(df_APS, df_AE)

# ver duplicadas de CO_CNES
df_merged %>%
  group_by(CO_CNES) %>%
  summarise(n = n()) %>%
  filter(n > 1) %>%
  arrange(desc(n))

# OK não tem duplicadas 

# quero criar um id_endereco para cada endereço único, ou seja, para cada combinação de NO_LOGRADOURO e NU_ENDERECO
# converte número inteiro em identificador de letras com k caracteres (base 26: A-Z)
num_para_letras <- function(n, k = 5) {
  sapply(n, function(x) {
    indices <- integer(k)
    for (i in k:1) {
      indices[i] <- x %% 26
      x <- x %/% 26
    }
    paste(LETTERS[indices + 1], collapse = "")
  })
}

df_merged <- df_merged |>
  dplyr::group_by(NO_LOGRADOURO, NU_ENDERECO) |>
  dplyr::mutate(id_endereco = num_para_letras(dplyr::cur_group_id() - 1, k = 5)) |>
  dplyr::ungroup()

# adicionar na tabela gercon_poa_censo_filtrado os id_endereco que correspondem a cada cnes
# quero criar 2 colunas no merge de id_endereco_cnes_solicitante 
# e id_endereco_cnesexecutante

gercon_poa_censo_filtrado_id <- gercon_poa_censo_filtrado %>%
  left_join(
    select(df_merged, CO_CNES, id_endereco),
    by = c("cnes_solicitante" = "CO_CNES")
  ) %>%
  rename(id_endereco_cnes_solicitante = id_endereco) %>%
  left_join(
    select(df_merged, CO_CNES, id_endereco),
    by = c("cnesexecutante" = "CO_CNES")
  ) %>%
  rename(id_endereco_cnesexecutante = id_endereco)

gercon_poa_censo_filtrado_id |> 
  mutate(igual = id_endereco_cnes_solicitante == id_endereco_cnesexecutante) |>
  group_by(igual) |>
  summarise(n = n()) |>
  arrange(desc(n))

#### API GOOGLE MAPS ####
# quero calcular a distância entre os endereços de cnes_solicitante e cnesexecutante 
# A função recebe id_endereco_cnes_solicitante e id_endereco_cnesexecutante
# consulta as informação de latitude e longitude de cada cnes no df_merged
# e depois consulta a API do Google Maps para calcular a distância entre os dois endereços
# A função retornar:
# distancia_km: menor distância a pé entre as rotas retornadas, em km.
# tempo_caminhando: tempo da rota a pé escolhida, em minutos.
# distancia_km_trans_publico: distância da rota de transporte escolhida, em km.
# tempo_desloc_trans_publico: menor tempo entre as alternativas de transporte, em minutos.
# quant_bus: número de trechos de ônibus da rota escolhida.
# baldiacoes: quant_bus − 1, com mínimo 0.

library(httr2)

# Configuração ----
arquivo_log_erros <- "log_erros_distancias.csv"
gmaps_key <- Sys.getenv("GOOGLE_MAPS_API_KEY")

# Log de erros ----
registrar_erro <- function(id_s, id_e, etapa, mensagem, arquivo = arquivo_log_erros) {
  linha <- tibble::tibble(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    id_endereco_cnes_solicitante = id_s,
    id_endereco_cnesexecutante = id_e,
    etapa = etapa,
    mensagem = mensagem
  )
  readr::write_csv(
    linha, arquivo,
    append = file.exists(arquivo),
    col_names = !file.exists(arquivo)
  )
  invisible(linha)
}

# Coordenadas por endereço ----
coords_lookup <- df_merged |>
  dplyr::mutate(
    lat = as.numeric(NU_LATITUDE),
    lon = as.numeric(NU_LONGITUDE)
  ) |>
  dplyr::filter(!is.na(lat), !is.na(lon)) |>
  dplyr::distinct(id_endereco, .keep_all = TRUE) |>
  dplyr::select(id_endereco, lat, lon)

# API Google Routes ----
rota_google <- function(origem, destino, modo, key = gmaps_key) {
  body <- list(
    origin = list(location = list(latLng = list(latitude = origem$lat, longitude = origem$lon))),
    destination = list(location = list(latLng = list(latitude = destino$lat, longitude = destino$lon))),
    travelMode = modo,
    computeAlternativeRoutes = TRUE,
    languageCode = "pt-BR",
    units = "METRIC"
  )
  resp <- request("https://routes.googleapis.com/directions/v2:computeRoutes") |>
    req_headers(
      `X-Goog-Api-Key` = key,
      `X-Goog-FieldMask` = paste(
        "routes.distanceMeters", "routes.duration",
        "routes.legs.steps.travelMode",
        "routes.legs.steps.transitDetails.transitLine.vehicle.type",
        sep = ","
      )
    ) |>
    req_body_json(body) |>
    req_error(is_error = \(resp) FALSE) |>
    req_perform()
  if (resp_status(resp) != 200) {
    corpo <- gsub("\\s+", " ", resp_body_string(resp))
    stop(sprintf("HTTP %s (%s): %s", resp_status(resp), modo, substr(corpo, 1, 300)), call. = FALSE)
  }
  resp_body_json(resp)$routes %||% list()
}

duracao_min <- function(x) as.numeric(sub("s$", "", x)) / 60

passos_onibus <- function(rota) {
  passos <- unlist(lapply(rota$legs, \(l) l$steps), recursive = FALSE)
  tipos <- vapply(passos, \(p) p$transitDetails$transitLine$vehicle$type %||% "", character(1))
  sum(tipos %in% c("BUS", "INTERCITY_BUS", "TROLLEYBUS"))
}

# Cálculo por par de endereços ----
distancia_cnes <- function(id_endereco_cnes_solicitante, id_endereco_cnesexecutante, lookup = coords_lookup) {
  id_s <- id_endereco_cnes_solicitante
  id_e <- id_endereco_cnesexecutante
  metricas <- tibble::tibble(
    distancia_km = NA_real_, tempo_caminhando = NA_real_,
    distancia_km_trans_publico = NA_real_, tempo_desloc_trans_publico = NA_real_,
    quant_bus = NA_integer_, baldiacoes = NA_integer_
  )

  tryCatch({
    if (identical(id_s, id_e)) {
      metricas[] <- 0
    } else {
      i_o <- match(id_s, lookup$id_endereco)
      i_d <- match(id_e, lookup$id_endereco)

      if (anyNA(c(i_o, i_d))) {
        registrar_erro(id_s, id_e, "coordenada", "ID sem coordenada em coords_lookup")
      } else {
        o <- lookup[i_o, ]
        d <- lookup[i_d, ]

        rotas_pe <- tryCatch(
          rota_google(o, d, "WALK"),
          error = \(e) { registrar_erro(id_s, id_e, "WALK", conditionMessage(e)); list() }
        )
        rotas_tp <- tryCatch(
          rota_google(o, d, "TRANSIT"),
          error = \(e) { registrar_erro(id_s, id_e, "TRANSIT", conditionMessage(e)); list() }
        )

        if (length(rotas_pe) > 0) {
          dist_pe <- vapply(rotas_pe, \(r) as.numeric(r$distanceMeters %||% 0), numeric(1))
          pe <- rotas_pe[[which.min(dist_pe)]]
          metricas$distancia_km <- min(dist_pe) / 1000
          metricas$tempo_caminhando <- duracao_min(pe$duration)
        }
        if (length(rotas_tp) > 0) {
          tempos <- vapply(rotas_tp, \(r) duracao_min(r$duration), numeric(1))
          tp <- rotas_tp[[which.min(tempos)]]
          n_bus <- as.integer(passos_onibus(tp))
          metricas$distancia_km_trans_publico <- as.numeric(tp$distanceMeters %||% 0) / 1000
          metricas$tempo_desloc_trans_publico <- min(tempos)
          metricas$quant_bus <- n_bus
          metricas$baldiacoes <- pmax(n_bus - 1L, 0L)
        }
      }
    }
  }, error = \(e) {
    registrar_erro(id_s, id_e, "inesperado", conditionMessage(e))
  })

  tibble::tibble(
    id_endereco_cnes_solicitante = id_s,
    id_endereco_cnesexecutante = id_e
  ) |>
    dplyr::bind_cols(metricas)
}

# Pares únicos ----
pares_unicos <- gercon_poa_censo_filtrado_id |>
  dplyr::filter(!is.na(id_endereco_cnes_solicitante), !is.na(id_endereco_cnesexecutante)) |>
  dplyr::distinct(id_endereco_cnes_solicitante, id_endereco_cnesexecutante)

# Execução ----
# Vou dividir em 2 

pares_unicos <- pares_unicos |>
  dplyr::mutate(lote = dplyr::if_else(dplyr::row_number() <= ceiling(dplyr::n() / 2), 1L, 2L))

pares_lote_1 <- pares_unicos |> dplyr::filter(lote == 1L) |> dplyr::select(-lote)
pares_lote_2 <- pares_unicos |> dplyr::filter(lote == 2L) |> dplyr::select(-lote)

c(total = nrow(pares_unicos), lote_1 = nrow(pares_lote_1), lote_2 = nrow(pares_lote_2))

# resultados_lote_1 <- purrr::pmap(pares_lote_1, distancia_cnes) |>
#    purrr::list_rbind()
# 
# saveRDS(resultados_lote_1, "resultados_distancias_lote_1.rds")

# Abri e  resultados_lote_1 <- readRDS("resultados_distancias_lote_1.rds")
resultados_lote_1 <- readRDS("resultados_distancias_lote_1.rds")

# verificar se todas as combinações de pares_lote_1 coluna 1 e coluna 2, estão em resultados_lote_1
# merge das colunas 1 e 2 do resultado_lote_1 com pares_lote_1, se houver algum NA, significa que não foi calculado a distância

merged <- dplyr::left_join(pares_lote_1, resultados_lote_1, by = c("id_endereco_cnes_solicitante", "id_endereco_cnesexecutante"))
if (any(is.na(merged$distancia_km))) {
  warning("Há combinações de pares_lote_1 que não foram calculadas em resultados_lote_1")
}

# verificar se tem algum NA em merged 
colSums(is.na(merged))
print(na_por_coluna)
if (any(na_por_coluna > 0)) {
  warning("Há valores NA em merged")
}

merge_teste_2 <- dplyr::left_join(pares_lote_2, resultados_lote_1, by = c("id_endereco_cnes_solicitante", "id_endereco_cnesexecutante"))

colSums(is.na(merge_teste_2))
print(na_por_coluna)
if (any(na_por_coluna > 0)) {
  warning("Há valores NA em merge_teste_2")
}

#
resultados_lote_2 <- purrr::pmap(pares_lote_2, distancia_cnes) |>
  purrr::list_rbind()
saveRDS(resultados_lote_2, "resultados_distancias_lote_2.rds")

# correção de erros ----
# ver NA em resultados_lote_1 e resultados_lote_2
colSums(is.na(resultados_lote_1))
colSums(is.na(resultados_lote_2))

# separar as linhas com NA em resultados_lote_1 e resultados_lote_2
na_lote_1 <- resultados_lote_1 |>
  dplyr::filter(dplyr::if_any(dplyr::everything(), is.na))

na_lote_2 <- resultados_lote_2 |>
  dplyr::filter(dplyr::if_any(dplyr::everything(), is.na))

na_lote_1
na_lote_2

merge_na <- dplyr::full_join(
  dplyr::select(na_lote_1, id_endereco_cnes_solicitante, id_endereco_cnesexecutante),
  dplyr::select(na_lote_2, id_endereco_cnes_solicitante, id_endereco_cnesexecutante),
  by = c("id_endereco_cnes_solicitante", "id_endereco_cnesexecutante")
)

# realizar nova consulta para merge_na
resultados_na <- purrr::pmap(merge_na, distancia_cnes) |>
  purrr::list_rbind()

colSums(is.na(resultados_na))

# Ficou sem a informação de:
# distancia_km_trans_publico | tempo_desloc_trans_publico | quant_bus | baldiacoes 
# Pq são outros municipios, e a API do Google Maps não retornou rotas de transporte público para esses pares de endereços.

## Unir resultados ----
resultados_distancias <- dplyr::bind_rows(resultados_lote_1, resultados_lote_2)

# confirmando as 17 linhas que ficaram sem a informação de transporte público
colSums(is.na(resultados_distancias))

# criar dataset final excluindo as linhas NA de resultados_distancias,
# e realizar o merge em gercon_poa_censo_filtrado_id, para adicionar as colunas de distancia_km, tempo_caminhando, distancia_km_trans_publico, tempo_desloc_trans_publico, quant_bus e baldiacoes
resultados_distancias_final <- resultados_distancias |>
  dplyr::filter(!is.na(distancia_km))

# realizar o merge em gercon_poa_censo_filtrado_id, para adicionar as colunas de distancia_km, tempo_caminhando, distancia_km_trans_publico, tempo_desloc_trans_publico, quant_bus e baldiacoes
gercon_poa_censo_filtrado_final <- gercon_poa_censo_filtrado_id |>
  dplyr::left_join(
    resultados_distancias_final,
    by = c("id_endereco_cnes_solicitante", "id_endereco_cnesexecutante")
  )

colSums(is.na(gercon_poa_censo_filtrado_final))

# salvar o dataset final em formato parquet
write_parquet(gercon_poa_censo_filtrado_final, "dados/processados/gercon_poa_censo_filtrado_final.parquet")
