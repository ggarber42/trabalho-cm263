# Distâncias entre CNES solicitante e executante (Google Routes API)
#
# Base de relacionamento: dados/distancias/base_distancias_onibus.parquet, uma linha por par de endereços.
# Por par são consultadas duas rotas: a pé (WALK) e de ônibus (TRANSIT, partida às 09:00 em data_referencia).
# calcular_distancias() consulta a API somente para pares sem resultado final
# (status ok, mesmo_local ou sem_rota). Falhas ficam em status e são retentadas na execução seguinte.
# montar_final() junta a base ao dataset de atendimento.

library(here)
library(arrow)
library(dplyr)
library(stringr)
library(httr2)
library(rlang)
library(readr)
library(purrr)
library(tibble)

# Configuração ----
dir_distancias <- here("dados", "distancias")
dir.create(dir_distancias, showWarnings = FALSE, recursive = TRUE)

arq_base <- file.path(dir_distancias, "base_distancias_onibus.parquet")
arquivo_final <- here("dados", "processados", "gercon_poa_censo_filtrado_final.parquet")
dir.create(dirname(arquivo_final), showWarnings = FALSE, recursive = TRUE)
gmaps_key <- Sys.getenv("GOOGLE_MAPS_API_KEY")

chaves <- c("id_endereco_cnes_solicitante", "id_endereco_cnesexecutante")
status_finais <- c("ok", "mesmo_local", "sem_rota")

# Data fixa para todas as execuções (deve ser futura). Brasil sem horário de verão desde 2019: -03:00.
data_referencia <- as.Date("2026-11-03")
partida_referencia <- paste0(format(data_referencia, "%Y-%m-%d"), "T09:00:00-03:00")

# Dados de atendimento ----
gercon_poa_censo_filtrado <- read_parquet(here("dados", "processados", "gercon_poa_censo_filtrado.parquet")) |>
  mutate(
    cnes_solicitante = str_pad(cnes_solicitante, width = 7, side = "left", pad = "0"),
    cnesexecutante = str_pad(cnesexecutante, width = 7, side = "left", pad = "0")
  )

# Endereços dos CNES ----
est <- read_csv2(
  here("tbEstabelecimento202609.csv"),
  locale = locale(encoding = "ISO-8859-1"),
  col_types = cols(.default = col_character())
) |>
  mutate(CO_CNES = str_pad(CO_CNES, width = 7, side = "left", pad = "0"))

cnes_usados <- unique(c(gercon_poa_censo_filtrado$cnes_solicitante, gercon_poa_censo_filtrado$cnesexecutante))
cnes_usados <- cnes_usados[!is.na(cnes_usados)]

enderecos_cnes <- est |>
  filter(CO_CNES %in% cnes_usados) |>
  transmute(
    CO_CNES,
    NO_LOGRADOURO = na_if(trimws(NO_LOGRADOURO), ""),
    NU_ENDERECO = na_if(trimws(NU_ENDERECO), ""),
    NO_BAIRRO = na_if(trimws(NO_BAIRRO), ""),
    CO_CEP = na_if(trimws(CO_CEP), ""),
    NU_LATITUDE,
    NU_LONGITUDE
  )

# Endereço incompleto impede a identificação do local
enderecos_incompletos <- enderecos_cnes |>
  filter(is.na(NO_LOGRADOURO) | is.na(NU_ENDERECO)) |>
  distinct(CO_CNES)
if (nrow(enderecos_incompletos) > 0) {
  stop(
    "CNES com logradouro ou número ausente em tbEstabelecimento202609.csv: ",
    paste(head(enderecos_incompletos$CO_CNES, 20), collapse = ", "),
    call. = FALSE
  )
}

# CNES ausentes do CSV (ex.: códigos de teste) são excluídos da base, com aviso.
cnes_ausentes <- setdiff(cnes_usados, enderecos_cnes$CO_CNES)
if (length(cnes_ausentes) > 0) {
  n_antes <- nrow(gercon_poa_censo_filtrado)
  gercon_poa_censo_filtrado <- gercon_poa_censo_filtrado |>
    filter(!cnes_solicitante %in% cnes_ausentes, !cnesexecutante %in% cnes_ausentes)
  warning(
    "CNES ausentes em tbEstabelecimento202609.csv excluídos da base: ",
    paste(cnes_ausentes, collapse = ", "), " (",
    n_antes - nrow(gercon_poa_censo_filtrado), " linhas removidas).",
    call. = FALSE
  )
  cnes_usados <- setdiff(cnes_usados, cnes_ausentes)
}

# Padronização do endereço ----
# Regras conservadoras: acentos, caixa, pontuação, prefixos de tipo de via e "S/N".
# Numerais por extenso (ex.: TRES x 3) NÃO são convertidos: o teste de proximidade
# mostrou que "TRES DE ABRIL" e "3 DE ABRIL" estão a ~880 m de distância.
padronizar_logradouro <- function(x) {
  x |>
    iconv(from = "UTF-8", to = "ASCII//TRANSLIT", sub = "") |>
    toupper() |>
    str_replace_all("[.,;:]", " ") |>
    str_squish() |>
    str_replace("^(R|RUA)\\s+", "RUA ") |>
    str_replace("^(AV|AVDA|AVENIDA)\\s+", "AVENIDA ") |>
    str_replace("^(EST|ESTRADA)\\s+", "ESTRADA ") |>
    str_replace("^(TRAV|TRAVESSA)\\s+", "TRAVESSA ") |>
    str_squish()
}

padronizar_numero <- function(x) {
  u <- x |>
    iconv(from = "UTF-8", to = "ASCII//TRANSLIT", sub = "") |>
    toupper() |>
    str_replace_all("[.\\s]", "") |>
    str_replace("^0+(?=\\d)", "")
  if_else(is.na(u) | u %in% c("", "S/N", "SN", "S/N°", "SEMNUMERO"), "SN", u)
}

enderecos_cnes <- enderecos_cnes |>
  mutate(
    logradouro_padrao = padronizar_logradouro(NO_LOGRADOURO),
    numero_padrao = padronizar_numero(NU_ENDERECO),
    chave_endereco = paste(logradouro_padrao, numero_padrao, NO_BAIRRO, CO_CEP, sep = "|"),
    # Campos ausentes viram texto vazio no endereço enviado à API
    numero_api = if_else(numero_padrao == "SN", "", coalesce(NU_ENDERECO, "")),
    bairro_api = coalesce(NO_BAIRRO, ""),
    cep_api = coalesce(CO_CEP, "")
  )

# Inspeção: original x padronizado, com bairro, CEP e coordenadas
enderecos_cnes |>
  select(
    CO_CNES, NO_LOGRADOURO, NU_ENDERECO, NO_BAIRRO, CO_CEP,
    NU_LATITUDE, NU_LONGITUDE, logradouro_padrao, numero_padrao, chave_endereco
  ) |>
  write_csv(file.path(dir_distancias, "enderecos_cnes_detalhe.csv"), na = "")

# Um CNES deve ter um único endereço
mapa_cnes <- enderecos_cnes |>
  distinct(CO_CNES, chave_endereco)
cnes_multiplos <- mapa_cnes |>
  count(CO_CNES) |>
  filter(n > 1)
if (nrow(cnes_multiplos) > 0) {
  stop(
    "CNES com mais de um endereço: ",
    paste(head(cnes_multiplos$CO_CNES, 20), collapse = ", "),
    call. = FALSE
  )
}

# Coordenadas válidas por endereço (faixa aproximada do Brasil)
coords_lookup <- enderecos_cnes |>
  mutate(
    lat = suppressWarnings(as.numeric(na_if(trimws(NU_LATITUDE), ""))),
    lon = suppressWarnings(as.numeric(na_if(trimws(NU_LONGITUDE), "")))
  ) |>
  filter(
    !is.na(lat), !is.na(lon),
    dplyr::between(lat, -35, 5), dplyr::between(lon, -75, -30)
  ) |>
  arrange(CO_CNES) |>
  distinct(chave_endereco, .keep_all = TRUE) |>
  transmute(
    id_endereco = chave_endereco, lat, lon,
    endereco_api = str_squish(str_replace_all(paste0(
      NO_LOGRADOURO,
      if_else(numero_api == "", "", paste0(", ", numero_api)),
      if_else(bairro_api == "", "", paste0(" - ", bairro_api)),
      ", Porto Alegre - RS, ", cep_api, ", Brasil"
    ), ",\\s*,", ","))
  )

cnes_sem_coord <- mapa_cnes |>
  filter(!chave_endereco %in% coords_lookup$id_endereco)
if (nrow(cnes_sem_coord) > 0) {
  stop(
    "CNES sem coordenada válida: ",
    paste(head(cnes_sem_coord$CO_CNES, 20), collapse = ", "),
    call. = FALSE
  )
}

gercon_id <- gercon_poa_censo_filtrado |>
  left_join(mapa_cnes, by = c("cnes_solicitante" = "CO_CNES")) |>
  rename(id_endereco_cnes_solicitante = chave_endereco) |>
  left_join(mapa_cnes, by = c("cnesexecutante" = "CO_CNES")) |>
  rename(id_endereco_cnesexecutante = chave_endereco)
stopifnot(nrow(gercon_id) == nrow(gercon_poa_censo_filtrado))

sem_id <- sum(is.na(gercon_id$id_endereco_cnes_solicitante) | is.na(gercon_id$id_endereco_cnesexecutante))
if (sem_id > 0) {
  stop(sem_id, " linhas sem id_endereco; verifique os CNES antes de calcular.", call. = FALSE)
}

pares_necessarios <- gercon_id |>
  distinct(across(all_of(chaves)))

# Base de relacionamento ----
base_vazia <- tibble(
  id_endereco_cnes_solicitante = character(),
  id_endereco_cnesexecutante = character(),
  status = character(),
  data_hora_partida = character(),
  consultado_em = character(),
  distancia_km = double(),
  tempo_caminhando = double(),
  distancia_km_trans_publico = double(),
  tempo_desloc_trans_publico = double(),
  quant_bus = integer(),
  baldiacoes = integer(),
  mensagem = character()
)

ler_base <- function() {
  if (!file.exists(arq_base)) return(base_vazia)
  read_parquet(arq_base)
}

# Grava em arquivo temporário e troca no final: interrupção não corrompe a base
salvar_base <- function(base) {
  tmp <- paste0(arq_base, ".tmp")
  write_parquet(base, tmp)
  file.rename(tmp, arq_base)
  invisible(base)
}

atualizar_base <- function(base, novos) {
  base |>
    anti_join(novos, by = chaves) |>
    bind_rows(novos)
}

pares_pendentes <- function(base) {
  pares_necessarios |>
    anti_join(filter(base, status %in% status_finais), by = chaves)
}

# API Google Routes ----
duracao_min <- function(x) {
  if (is.null(x)) return(NA_real_)
  suppressWarnings(as.numeric(sub("s$", "", x))) / 60
}

passos_onibus <- function(rota) {
  passos <- unlist(lapply(rota$legs %||% list(), \(l) l$steps), recursive = FALSE)
  tipos <- vapply(passos, \(p) p$transitDetails$transitLine$vehicle$type %||% "", character(1))
  as.integer(sum(tipos %in% c("BUS", "INTERCITY_BUS", "TROLLEYBUS")))
}

# modo: "WALK" ou "TRANSIT" (somente ônibus, com partida fixa)
req_rota <- function(origem, destino, modo, partida = NULL, key = gmaps_key) {
  body <- list(
    origin = list(address = origem$endereco_api),
    destination = list(address = destino$endereco_api),
    travelMode = modo,
    computeAlternativeRoutes = TRUE,
    languageCode = "pt-BR",
    units = "METRIC"
  )
  if (modo == "TRANSIT") {
    body$departureTime <- partida
    body$transitPreferences <- list(allowedTravelModes = list("BUS"))
  }
  request("https://routes.googleapis.com/directions/v2:computeRoutes") |>
    req_headers(
      `X-Goog-Api-Key` = key,
      `X-Goog-FieldMask` = paste(
        "routes.duration", "routes.distanceMeters",
        "routes.legs.steps.travelMode",
        "routes.legs.steps.transitDetails.transitLine.vehicle.type",
        sep = ","
      )
    ) |>
    req_body_json(body) |>
    req_timeout(60) |>
    req_retry(
      max_tries = 3,
      is_transient = \(resp) resp_status(resp) %in% c(429, 500, 502, 503, 504),
      backoff = \(i) 2^i
    ) |>
    req_error(is_error = \(resp) FALSE)
}

# Teste da chave com uma rota a pé e uma de ônibus antes de qualquer lote grande
verificar_chave <- function() {
  if (!nzchar(gmaps_key)) {
    stop("GOOGLE_MAPS_API_KEY não definida. Configure em ~/.Renviron e reinicie o R.", call. = FALSE)
  }
  a <- coords_lookup[1, ]
  b <- coords_lookup |>
    filter(id_endereco != a$id_endereco) |>
    slice(1)
  if (nrow(b) == 0) {
    stop("São necessários ao menos dois endereços com coordenada para testar a chave.", call. = FALSE)
  }
  for (modo in c("WALK", "TRANSIT")) {
    resp <- req_rota(a, b, modo, partida_referencia) |> req_perform()
    if (resp_status(resp) != 200) {
      stop(
        "Chave da API falhou (", modo, "): HTTP ", resp_status(resp), " - ",
        substr(gsub("\\s+", " ", resp_body_string(resp)), 1, 300),
        call. = FALSE
      )
    }
  }
  message("Chave da API válida (WALK e TRANSIT).")
  invisible(TRUE)
}

# Retorna NULL se a resposta é utilizável; senão, status e mensagem do erro
classificar_erro <- function(resp) {
  if (inherits(resp, "error")) {
    return(list(status = "erro_conexao", msg = substr(conditionMessage(resp), 1, 500)))
  }
  st <- resp_status(resp)
  if (st == 200) return(NULL)
  msg <- substr(gsub("\\s+", " ", resp_body_string(resp)), 1, 500)
  list(
    status = if (st %in% c(429, 500, 502, 503, 504)) "erro_transitorio" else "erro_api",
    msg = paste0("HTTP ", st, ": ", msg)
  )
}

# Converte as respostas a pé e de ônibus de um par em uma linha da base
interpretar_resposta <- function(resp_pe, resp_tp, par, partida) {
  base <- par |>
    mutate(
      data_hora_partida = partida,
      consultado_em = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      distancia_km = NA_real_,
      tempo_caminhando = NA_real_,
      distancia_km_trans_publico = NA_real_,
      tempo_desloc_trans_publico = NA_real_,
      quant_bus = NA_integer_,
      baldiacoes = NA_integer_,
      mensagem = NA_character_
    )

  erro <- classificar_erro(resp_tp) %||% classificar_erro(resp_pe)
  if (!is.null(erro)) {
    return(mutate(base, status = erro$status, mensagem = erro$msg))
  }

  rotas_pe <- resp_body_json(resp_pe)$routes %||% list()
  if (length(rotas_pe) > 0) {
    dist_pe <- vapply(rotas_pe, \(r) as.numeric(r$distanceMeters %||% NA_real_), numeric(1))
    tempo_pe <- vapply(rotas_pe, \(r) duracao_min(r$duration), numeric(1))
    if (any(!is.na(dist_pe))) {
      i <- which.min(dist_pe)
      base$distancia_km <- dist_pe[i] / 1000
      base$tempo_caminhando <- tempo_pe[i]
    }
  }

  rotas_tp <- resp_body_json(resp_tp)$routes %||% list()
  if (length(rotas_tp) == 0) return(mutate(base, status = "sem_rota"))

  tempos <- vapply(rotas_tp, \(r) duracao_min(r$duration), numeric(1))
  if (all(is.na(tempos))) {
    return(mutate(base, status = "erro_api", mensagem = "rotas de ônibus sem duração"))
  }
  rota <- rotas_tp[[which.min(tempos)]]
  n_bus <- passos_onibus(rota)
  if (n_bus == 0L) return(mutate(base, status = "sem_rota"))

  mutate(
    base,
    status = "ok",
    distancia_km_trans_publico = as.numeric(rota$distanceMeters %||% NA) / 1000,
    tempo_desloc_trans_publico = min(tempos, na.rm = TRUE),
    quant_bus = n_bus,
    baldiacoes = pmax(n_bus - 1L, 0L)
  )
}

# Requisições em paralelo (a pé e ônibus) para um bloco de pares
processar_bloco <- function(pares) {
  origens <- coords_lookup[match(pares$id_endereco_cnes_solicitante, coords_lookup$id_endereco), ]
  destinos <- coords_lookup[match(pares$id_endereco_cnesexecutante, coords_lookup$id_endereco), ]
  pares_o <- split(origens, seq_len(nrow(origens)))
  pares_d <- split(destinos, seq_len(nrow(destinos)))

  reqs_pe <- map2(pares_o, pares_d, \(o, d) req_rota(o, d, "WALK"))
  reqs_tp <- map2(pares_o, pares_d, \(o, d) req_rota(o, d, "TRANSIT", partida_referencia))
  resps_pe <- req_perform_parallel(reqs_pe, max_active = 4, on_error = "continue", progress = FALSE)
  resps_tp <- req_perform_parallel(reqs_tp, max_active = 4, on_error = "continue", progress = FALSE)

  map(seq_len(nrow(pares)), \(i) {
    interpretar_resposta(resps_pe[[i]], resps_tp[[i]], pares[i, ], partida_referencia)
  }) |>
    list_rbind()
}

# Cálculo ----
# Pares com endereço igual recebem zero sem consulta. Demais pares pendentes são consultados em blocos,
# com a base salva após cada bloco. limite restringe o número de pares por execução.
calcular_distancias <- function(tamanho_bloco = 200, limite = NULL) {
  if (data_referencia <= Sys.Date()) {
    stop("data_referencia deve ser uma data futura para a API aceitar a partida.", call. = FALSE)
  }
  base <- ler_base()

  mesmo_local <- pares_pendentes(base) |>
    filter(id_endereco_cnes_solicitante == id_endereco_cnesexecutante)
  if (nrow(mesmo_local) > 0) {
    novos <- mesmo_local |>
      mutate(
        status = "mesmo_local",
        data_hora_partida = partida_referencia,
        consultado_em = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        distancia_km = 0,
        tempo_caminhando = 0,
        distancia_km_trans_publico = 0,
        tempo_desloc_trans_publico = 0,
        quant_bus = 0L,
        baldiacoes = 0L,
        mensagem = NA_character_
      )
    base <- atualizar_base(base, novos)
    salvar_base(base)
  }

  pendentes <- pares_pendentes(base) |>
    filter(id_endereco_cnes_solicitante != id_endereco_cnesexecutante)
  if (!is.null(limite)) pendentes <- head(pendentes, limite)
  n <- nrow(pendentes)
  message("Pares a consultar na API: ", n)
  if (n == 0) return(invisible(base))

  verificar_chave()

  blocos <- split(seq_len(n), ceiling(seq_len(n) / tamanho_bloco))
  for (i in seq_along(blocos)) {
    resultados <- processar_bloco(pendentes[blocos[[i]], ])
    base <- atualizar_base(base, resultados)
    salvar_base(base)

    tab <- table(resultados$status)
    message(
      "Bloco ", i, " de ", length(blocos), ": ",
      paste(names(tab), tab, sep = "=", collapse = ", ")
    )

    if (any(grepl("^HTTP (401|403):", resultados$mensagem))) {
      stop("Chave rejeitada pela API (HTTP 401/403). Progresso salvo.", call. = FALSE)
    }
    falhas <- mean(resultados$status %in% c("erro_transitorio", "erro_conexao"))
    if (falhas > 0.5) {
      stop("Mais de 50% de falhas no bloco. Progresso salvo; execute novamente depois.", call. = FALSE)
    }
  }
  invisible(base)
}

# Resumo ----
resumo_status <- function() {
  ler_base() |>
    semi_join(pares_necessarios, by = chaves) |>
    count(status, name = "n_pares") |>
    arrange(desc(n_pares))
}

# Junta distâncias ao atendimento ----
# Junta as distâncias da base ao objeto de atendimento (uma linha por registro).
# Pares sem resultado final ficam como NA em todas as métricas e no status.
juntar_distancias <- function(dados = gercon_id) {
  distancias <- ler_base() |>
    semi_join(pares_necessarios, by = chaves) |>
    select(
      all_of(chaves),
      status_distancia = status,
      data_hora_partida,
      distancia_km,
      tempo_caminhando,
      distancia_km_trans_publico,
      tempo_desloc_trans_publico,
      quant_bus,
      baldiacoes
    )

  final <- dados |>
    left_join(distancias, by = chaves, relationship = "many-to-one")
  if (nrow(final) != nrow(dados)) {
    stop("O join alterou o número de linhas do dataset.", call. = FALSE)
  }
  final
}

# Montagem final ----
# Só executa quando todos os pares tiverem resultado final.
montar_final <- function() {
  faltando <- pares_pendentes(ler_base())
  if (nrow(faltando) > 0) {
    stop(nrow(faltando), " pares sem resultado final; execute calcular_distancias() novamente.", call. = FALSE)
  }

  final <- juntar_distancias(gercon_id)

  write_parquet(final, arquivo_final)
  message("Dataset final gravado em ", arquivo_final, " com ", nrow(final), " linhas.")
  invisible(final)
}

# Uso:
# calcular_distancias()              # consulta somente pares sem resultado final; pode ser repetido
# calcular_distancias(limite = 500)  # processa parte dos pares pendentes
# resumo_status()
# montar_final()


### Qualificar os dados do arquivo base_distancias_onibus.parquet
# A base de distâncias pode ser qualificada com base em inspeção visual, conhecimento local
# e análise de consistência.

dados<-read_parquet("dados/distancias/base_distancias_onibus.parquet")

# calcular NULL na variavels distancia_km pela variavel status
dados |> 
group_by(status) |> 
  summarise(n = n(), n_distancia_km_null = sum(is.na(distancia_km))) |> 
  arrange(desc(n)) |> 
  print(n=Inf)

# vericar subgrupo sem_rota para idenficar o motivo dos NULL da variavel distancia_km
subgrupo<-dados |> 
  filter(status == "sem_rota", is.na(distancia_km)) 

# os 23 subgrupos na verdade são o mesmo endereço, com isso vou colocar distancia_km = 0, tempo_caminhando = 0, distancia_km_trans_publico = 0, tempo_desloc_trans_publico = 0, quant_bus = 0, baldiacoes = 0
summary(dados$distancia_km)

dados <- dados |>
  mutate(
    distancia_km = if_else(status == "sem_rota" & is.na(distancia_km), 0, distancia_km),
    tempo_caminhando = if_else(status == "sem_rota" & is.na(tempo_caminhando), 0, tempo_caminhando),
    distancia_km_trans_publico = if_else(status == "sem_rota" & is.na(distancia_km_trans_publico), 0, distancia_km_trans_publico), 
    tempo_desloc_trans_publico = if_else(status == "sem_rota" & is.na(tempo_desloc_trans_publico), 0, tempo_desloc_trans_publico),
    quant_bus = if_else(status == "sem_rota" & is.na(quant_bus), 0, quant_bus),
    baldiacoes = if_else(status == "sem_rota" & is.na(baldiacoes), 0, baldiacoes)
  )

summary(dados$distancia_km)

# salvar o arquivo base_distancias_onibus.parquet qualificado
# write_parquet(dados, "dados/distancias/base_distancias_onibus.parquet")
 
