# Teste de estacionariedade de segunda ordem por wavelets (Nason 2013), pacote locits.
# Opcional: se o pacote não estiver instalado, o pipeline usa ADF + KPSS e registra isso.
# Uso: Rscript r/teste_wavelet.R <entrada.csv> <saida.csv>
#   entrada: colunas serie, semana, y (só treino)
#   saida:   serie, n_usado, p_wavelet
# DECISÃO: locits exige comprimento potência de 2; usam-se as últimas 2^k semanas do treino.
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("locits", quietly = TRUE)) {
  message("Pacote locits indisponível: teste de wavelets não executado.")
  quit(status = 0)
}
d <- read.csv(args[1], colClasses = c(serie = "character"))
res <- do.call(rbind, lapply(split(d, d$serie), function(s) {
  s <- s[order(s$semana), ]
  n <- 2^floor(log2(nrow(s)))
  y <- tail(s$y, n)
  p <- NA_real_
  if (n >= 64 && sd(y) > 0) {
    t <- tryCatch(locits::hwtos2(y), error = function(e) NULL)
    # hwtos2 já aplica a correção para os múltiplos coeficientes; nreject > 0 = não estacionária.
    # A VERIFICAR ao instalar o locits: nome do campo `nreject` (não testado nesta máquina).
    # Gravamos um indicador (0 = rejeita, 1 = não rejeita) no lugar de um valor-p.
    if (!is.null(t)) p <- if (t$nreject > 0) 0 else 1
  }
  data.frame(serie = s$serie[1], n_usado = n, p_wavelet = p)
}))
write.csv(res, args[2], row.names = FALSE)
