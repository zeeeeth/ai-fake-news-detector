import { useMemo, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import './App.css'
import apiService from './services/api'
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom'
import { AppBar, Toolbar, Typography, Button, Box, Paper } from '@mui/material'
import CheckArticle from './components/checkArticle'
import CheckClaim from './components/checkClaim'
import Home from './components/home'

type Prediction = {
  label: string
  confidence: number
  probabilities: { [key: string]: number }
}

function TopNav() {
  const location = useLocation()
  const active = (path: string) => location.pathname === path

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        bgcolor: '#0f1115',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      <Toolbar sx={{ gap: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          AI Fake News Detector
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, ml: 1 }}>
          <Button
            component={Link}
            to="/claims"
            variant={active('/claims') ? 'contained' : 'outlined'}
            sx={{ textTransform: 'none' }}
          >
            Check Claim
          </Button>
          <Button
            component={Link}
            to="/articles"
            variant={active('/articles') ? 'contained' : 'outlined'}
            sx={{ textTransform: 'none' }}
          >
            Check Article
          </Button>
          <Button
            component={Link}
            to="/"
            variant={active('/') ? 'contained' : 'outlined'}
            sx={{ textTransform: 'none' }}
          >
            Home
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  )
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x))
}

function asTrueFalseProbs(prediction: Prediction | null) {
  if (!prediction) return null

  const probs = prediction.probabilities || {}

  const pick = (key: string) => {
    const direct = probs[key]
    if (typeof direct === 'number') return direct
    const cap = key.charAt(0).toUpperCase() + key.slice(1)
    const caps = key.toUpperCase()
    const low = key.toLowerCase()
    const v = probs[cap] ?? probs[caps] ?? probs[low]
    return typeof v === 'number' ? v : undefined
  }

  // 1) Prefer explicit probs: true/false
  const t1 = pick('true')
  const f1 = pick('false')
  if (typeof t1 === 'number' && typeof f1 === 'number') {
    const sum = t1 + f1
    const t = sum > 0 ? t1 / sum : 0.5
    return { trueProb: clamp01(t), falseProb: clamp01(1 - t) }
  }

  // 2) Common ML naming: real/fake (treat real as True, fake as False)
  const real = pick('real')
  const fake = pick('fake')
  if (typeof real === 'number' && typeof fake === 'number') {
    const sum = real + fake
    const t = sum > 0 ? real / sum : 0.5
    return { trueProb: clamp01(t), falseProb: clamp01(1 - t) }
  }

  // 3) Fallback: label + confidence (support true/false and real/fake)
  if (typeof prediction.confidence === 'number' && typeof prediction.label === 'string') {
    const conf = clamp01(prediction.confidence)
    const label = prediction.label.toLowerCase()

    const labelMeansTrue = label === 'true' || label === 'real'
    const labelMeansFalse = label === 'false' || label === 'fake'

    if (labelMeansTrue) return { trueProb: conf, falseProb: clamp01(1 - conf) }
    if (labelMeansFalse) return { trueProb: clamp01(1 - conf), falseProb: conf }
  }

  // 4) Default neutral
  return { trueProb: 0.5, falseProb: 0.5 }
}

function Dot({ color }: { color: string }) {
  return (
    <Box
      component="span"
      sx={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        bgcolor: color,
        display: 'inline-block',
        flex: '0 0 auto',
      }}
    />
  )
}


function PredictionPanel({
  prediction,
  mode,
  ringKey,
}: {
  prediction: Prediction | null
  mode: 'article' | 'claim'
  ringKey: number
}) {
  // ---------- helpers ----------
  const prettyLabel = (s: string) =>
    String(s || '')
      .replace(/_/g, ' ')
      .replace(/-/g, ' ')
      .trim()
      .split(/\s+/)
      .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
      .join(' ')

  const clamp01Local = (x: number) => Math.max(0, Math.min(1, x))

  const tf = useMemo(() => {
    if (mode !== 'article') return null
    return asTrueFalseProbs(prediction)
  }, [mode, prediction])

  const top = useMemo(() => {
    if (mode !== 'claim') return null
    if (!prediction) return null

    const probsObj = prediction.probabilities || {}
    const entries = Object.entries(probsObj)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
      .map(([k, v]) => ({ label: String(k), prob: Math.max(0, Number(v)) }))
      .filter((x) => x.prob > 0)

    if (entries.length === 0 && typeof prediction.label === 'string' && typeof prediction.confidence === 'number') {
      const conf = clamp01(prediction.confidence)
      return {
        items: [{ label: prediction.label, prob: conf }],
        otherProb: clamp01(1 - conf),
      }
    }

    const sum = entries.reduce((a, b) => a + b.prob, 0)
    const norm = sum > 0 ? entries.map((e) => ({ ...e, prob: e.prob / sum })) : []

    norm.sort((a, b) => b.prob - a.prob)
    const items = norm.slice(0, 3)
    const topSum = items.reduce((a, b) => a + b.prob, 0)
    const otherProb = clamp01(1 - topSum)

    return { items, otherProb }
  }, [mode, prediction])


  // ---------- ARTICLE MODE (binary) ----------
  if (mode === 'article') {
    const hasPrediction = !!prediction && !!tf

    const predictedIsTrue = hasPrediction ? tf!.trueProb >= 0.5 : true
    const predictedLabel = !hasPrediction ? '?' : predictedIsTrue ? 'True' : 'False'

    const predictedProb = !hasPrediction ? 0 : predictedIsTrue ? tf!.trueProb : tf!.falseProb
    const otherProb = !hasPrediction ? 0 : predictedIsTrue ? tf!.falseProb : tf!.trueProb

    const predictedPct = Math.round(predictedProb * 1000) / 10

    const truePct = hasPrediction ? Math.round(tf!.trueProb * 1000) / 10 : 0
    const falsePct = hasPrediction ? Math.round(tf!.falseProb * 1000) / 10 : 0

    // Ring colours (still “predicted first, other fills rest”)
    const primaryColor = predictedIsTrue ? '#2e7d32' : '#c62828'
    const secondaryColor = predictedIsTrue ? '#c62828' : '#2e7d32'

    const size = 240
    const stroke = 18
    const r = (size - stroke) / 2
    const c = 2 * Math.PI * r

    const primaryLen = c * clamp01Local(predictedProb)
    const secondaryLen = c * clamp01Local(otherProb)

    const animPrimary = `dashPrimary_article_${ringKey}`
    const animSecondary = `dashSecondary_article_${ringKey}`

    return (
      <Paper
        variant="outlined"
        sx={{
          p: 3,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: '#e0e0e0',
          color: '#111',
          borderColor: 'rgba(0,0,0,0.18)',
          borderRadius: 3,
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
          boxSizing: 'border-box',
          overflowX: 'hidden',
        }}
      >
        <Typography variant="h6" sx={{ fontWeight: 800 }}>
          Prediction
        </Typography>

        <Typography variant="body2" sx={{ mt: 0.5, color: 'rgba(0,0,0,0.65)' }}>
          {!hasPrediction ? 'Submit an article…' : ''}
        </Typography>

        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2 }}>
          {!hasPrediction ? (
            <Box sx={{ position: 'relative' }}>
              <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                <Box sx={{ width: size, height: size, position: 'relative' }}>
                  <svg width={size} height={size} style={{ display: 'block' }}>
                    <g transform={`translate(${size / 2},${size / 2}) rotate(-90)`}>
                      <circle r={r} cx={0} cy={0} fill="none" stroke="#bdbdbd" strokeWidth={stroke} />
                    </g>
                  </svg>
                </Box>
              </Box>
              <Box
                sx={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                }}
              >
                <Typography sx={{ fontWeight: 900, fontSize: 56, lineHeight: 1 }}>
                  ?
                </Typography>
              </Box>
            </Box>
          ) : (
            <Box sx={{ position: 'relative' }}>
              <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                {/* key forces remount each submit */}
                <Box key={`article-${ringKey}`} sx={{ width: size, height: size, position: 'relative' }}>
                  <svg width={size} height={size} style={{ display: 'block' }}>
                    <g transform={`translate(${size / 2},${size / 2}) rotate(-90)`}>
                      <circle r={r} cx={0} cy={0} fill="none" stroke="#bdbdbd" strokeWidth={stroke} />

                      <circle
                        r={r}
                        cx={0}
                        cy={0}
                        fill="none"
                        stroke={primaryColor}
                        strokeWidth={stroke}
                        strokeLinecap="round"
                        strokeDasharray={`0 ${c}`}
                        style={{ animation: `${animPrimary} 750ms ease-out forwards` }}
                      />

                      <circle
                        r={r}
                        cx={0}
                        cy={0}
                        fill="none"
                        stroke={secondaryColor}
                        strokeWidth={stroke}
                        strokeLinecap="round"
                        strokeDasharray={`0 ${c}`}
                        strokeDashoffset={-primaryLen}
                        style={{
                          animation: `${animSecondary} 750ms ease-out forwards`,
                          animationDelay: '750ms',
                        }}
                      />
                    </g>

                    <style>{`
                      @keyframes ${animPrimary} {
                        from { stroke-dasharray: 0 ${c}; }
                        to { stroke-dasharray: ${primaryLen} ${c - primaryLen}; }
                      }
                      @keyframes ${animSecondary} {
                        from { stroke-dasharray: 0 ${c}; }
                        to { stroke-dasharray: ${secondaryLen} ${c - secondaryLen}; }
                      }
                    `}</style>
                  </svg>
                </Box>
              </Box>

              <Box
                sx={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  px: 2,
                }}
              >
                <Typography sx={{ fontWeight: 900, fontSize: 42, lineHeight: 1.1 }}>
                  {predictedLabel}
                </Typography>
                <Typography sx={{ fontWeight: 700, fontSize: 18, color: 'rgba(0,0,0,0.7)' }}>
                  {predictedPct.toFixed(1)}%
                </Typography>
              </Box>
            </Box>
          )}
        </Box>

        {hasPrediction && (
          <Box sx={{ mt: 1 }}>
            {/* Bottom legend with coloured dots */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
              <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
                <Dot color="#2e7d32" />
                <Typography variant="body2" component="span" sx={{ color: 'rgba(0,0,0,0.75)' }}>
                  True: <b>{truePct.toFixed(1)}%</b>
                </Typography>
              </Box>

              <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
                <Dot color="#c62828" />
                <Typography variant="body2" component="span" sx={{ color: 'rgba(0,0,0,0.75)' }}>
                  False: <b>{falsePct.toFixed(1)}%</b>
                </Typography>
              </Box>
            </Box>

            {/* (Optional) keep your previous text line if you want; otherwise remove this */}
            {/* <Typography variant="body2" sx={{ mt: 1, color: 'rgba(0,0,0,0.55)' }}>
              Predicted {predictedLabel} ({predictedPct.toFixed(1)}%) · Other {otherLabel} ({otherPct.toFixed(1)}%)
            </Typography> */}
          </Box>
        )}
      </Paper>
    )
  }

  // ---------- CLAIM MODE (top 3 + Other) ----------
  const hasPrediction = !!prediction && !!top

  const size = 240
  const stroke = 18
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r

  const palette = ['#1565c0', '#2e7d32', '#c62828'] // top1/top2/top3
  const otherColor = '#9e9e9e'

  const segs = hasPrediction
    ? [
        ...top!.items.map((it, i) => ({
          label: it.label,
          pretty: prettyLabel(it.label),
          prob: clamp01Local(it.prob),
          color: palette[i] || '#6d4c41',
        })),
        ...(top!.otherProb > 0.0001
          ? [{ label: 'other', pretty: 'Other', prob: clamp01Local(top!.otherProb), color: otherColor }]
          : []),
      ]
    : []

  const best = hasPrediction && segs.length > 0 ? segs[0] : null

  const segLens = segs.map((s) => c * s.prob)
  const cumLens: number[] = []
  let running = 0
  for (let i = 0; i < segLens.length; i++) {
    cumLens.push(running)
    running += segLens[i]
  }

  // legend items: top3 + other (if present)
  const legendSegs = hasPrediction
    ? (() => {
        const first3 = segs.slice(0, 3)
        const other = segs.find((s) => s.pretty === 'Other')
        return other ? [...first3, other] : first3
      })()
    : []

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 3,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: '#e0e0e0',
        color: '#111',
        borderColor: 'rgba(0,0,0,0.18)',
        borderRadius: 3,
        width: '100%',
        maxWidth: '100%',
        minWidth: 0,
        boxSizing: 'border-box',
        overflowX: 'hidden',
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 800 }}>
        Prediction
      </Typography>

      <Typography variant="body2" sx={{ mt: 0.5, color: 'rgba(0,0,0,0.65)' }}>
        {!hasPrediction ? 'Submit a claim…' : 'Your results:'}
      </Typography>

      <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2 }}>
        {!hasPrediction ? (
          <Box sx={{ position: 'relative' }}>
            <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              <Box sx={{ width: size, height: size, position: 'relative' }}>
                <svg width={size} height={size} style={{ display: 'block' }}>
                  <g transform={`translate(${size / 2},${size / 2}) rotate(-90)`}>
                    <circle r={r} cx={0} cy={0} fill="none" stroke="#bdbdbd" strokeWidth={stroke} />
                  </g>
                </svg>
              </Box>
            </Box>
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
              }}
            >
              <Typography sx={{ fontWeight: 900, fontSize: 56, lineHeight: 1 }}>
                ?
              </Typography>
            </Box>
          </Box>
        ) : (
          <Box sx={{ position: 'relative' }}>
            <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              {/* key forces remount each submit */}
              <Box key={`claim-${ringKey}`} sx={{ width: size, height: size, position: 'relative' }}>
                <svg width={size} height={size} style={{ display: 'block' }}>
                  <g transform={`translate(${size / 2},${size / 2}) rotate(-90)`}>
                    <circle r={r} cx={0} cy={0} fill="none" stroke="#bdbdbd" strokeWidth={stroke} />

                    {segs.map((s, i) => {
                      const offset = -cumLens[i]
                      const anim = `dashClaim_${ringKey}_${i}`

                      return (
                        <circle
                          key={`${ringKey}-${s.label}-${i}`}
                          r={r}
                          cx={0}
                          cy={0}
                          fill="none"
                          stroke={s.color}
                          strokeWidth={stroke}
                          strokeLinecap="round"
                          strokeDasharray={`0 ${c}`}
                          strokeDashoffset={offset}
                          style={{
                            animation: `${anim} 550ms ease-out forwards`,
                            animationDelay: `${i * 250}ms`,
                          }}
                        />
                      )
                    })}
                  </g>

                  <style>{`
                    ${segs
                      .map((_, i) => {
                        const len = segLens[i]
                        return `
                          @keyframes dashClaim_${ringKey}_${i} {
                            from { stroke-dasharray: 0 ${c}; }
                            to { stroke-dasharray: ${len} ${c - len}; }
                          }
                        `
                      })
                      .join('\n')}
                  `}</style>
                </svg>
              </Box>
            </Box>

            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                px: 2,
              }}
            >
              <Typography sx={{ fontWeight: 900, fontSize: 34, lineHeight: 1.15 }}>
                {best ? best.pretty : '?'}
              </Typography>
              <Typography sx={{ fontWeight: 700, fontSize: 18, color: 'rgba(0,0,0,0.7)' }}>
                {best ? (best.prob * 100).toFixed(1) : '0.0'}%
              </Typography>
            </Box>
          </Box>
        )}
      </Box>

      {hasPrediction && (
        <Box sx={{ mt: 1 }}>
          {/* Bottom legend with coloured dots */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            {legendSegs.map((s) => (
              <Box key={s.pretty} sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
                <Dot color={s.color} />
                <Typography variant="body2" component="span" sx={{ color: 'rgba(0,0,0,0.75)' }}>
                  {s.pretty}: <b>{(s.prob * 100).toFixed(1)}%</b>
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Paper>
  )
}



function TwoColumnPage({ left, right }: { left: ReactNode; right: ReactNode }) {
  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: '100vw',
        boxSizing: 'border-box',
        overflowX: 'hidden',
        px: { xs: 2, md: 3 },
        py: 4,
      }}
    >
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
          gap: { xs: 2, md: 2 },
          alignItems: 'stretch',
          width: '100%',
          maxWidth: '100%',
          boxSizing: 'border-box',
        }}
      >
        <Box sx={{ minWidth: 0, maxWidth: '100%', height: '100%' }}>{left}</Box>
        <Box sx={{ minWidth: 0, maxWidth: '100%', height: '100%' }}>{right}</Box>
      </Box>
    </Box>
  )
}

const App = () => {
  const [article, setArticle] = useState('')
  const [claim, setClaim] = useState('')

  // keep predictions separate so pages don’t stomp each other
  const [articlePrediction, setArticlePrediction] = useState<Prediction | null>(null)
  const [claimPrediction, setClaimPrediction] = useState<Prediction | null>(null)
  const [articleRingKey, setArticleRingKey] = useState(0)
  const [claimRingKey, setClaimRingKey] = useState(0)

  const submitArticle = async (e: FormEvent) => {
    e.preventDefault()
    console.log('[submitArticle] fired', { length: article.length })

    try {
      const pred = await apiService.predictArticle(article)
      console.log('[submitArticle] response', pred)
      setArticlePrediction(pred)
      setArticleRingKey((k) => k + 1) // <- force ring remount
    } catch (err) {
      console.error('[submitArticle] FAILED', err)
    }
  }

  const submitClaim = async (e: FormEvent) => {
    e.preventDefault()
    console.log('[submitClaim] fired', { length: claim.length })

    try {
      const pred = await apiService.predictClaim(claim)
      console.log('[submitClaim] response', pred)
      setClaimPrediction(pred)
      setClaimRingKey((k) => k + 1) // <- force ring remount
    } catch (err) {
      console.error('[submitClaim] FAILED', err)
    }
  }


  return (
    <Router>
      <Box sx={{ minHeight: '100vh' }}>
        <TopNav />

        <Routes>
          <Route path="/" element={<Home />} />

          <Route
            path="/articles"
            element={
              <TwoColumnPage
                left={<CheckArticle submitArticle={submitArticle} article={article} setArticle={setArticle} />}
                right={<PredictionPanel prediction={articlePrediction} mode="article" ringKey={articleRingKey}/>}
              />
            }
          />

          <Route
            path="/claims"
            element={
              <TwoColumnPage
                left={<CheckClaim submitClaim={submitClaim} claim={claim} setClaim={setClaim} />}
                right={<PredictionPanel prediction={claimPrediction} mode="claim" ringKey={claimRingKey}/>}
              />
            }
          />
        </Routes>
      </Box>
    </Router>
  )
}

export default App
