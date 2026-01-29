import { Box, Typography, Paper, Divider, Link } from '@mui/material'

const Home = () => {
  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: { xs: 2, md: 3 },
        py: { xs: 3, md: 5 },
      }}
    >
      {/* Outer box */}
      <Paper
        variant="outlined"
        sx={{
          width: '100%',
          maxWidth: 980,
          p: { xs: 3, md: 4 },
          borderRadius: 4,
          bgcolor: '#e0e0e0',
          color: '#111',
          borderColor: 'rgba(0,0,0,0.18)',
          boxShadow: '0 14px 40px rgba(0,0,0,0.35)',
        }}
      >
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Typography
            variant="h3"
            sx={{
              fontWeight: 900,
              letterSpacing: -0.6,
              fontSize: { xs: 30, sm: 38, md: 44 },
              lineHeight: 1.05,
            }}
          >
            AI Fake News Detector
          </Typography>

          <Typography
            sx={{
              mt: 1,
              color: 'rgba(0,0,0,0.70)',
              fontSize: { xs: 14, sm: 16 },
              maxWidth: 760,
              mx: 'auto',
              lineHeight: 1.6,
            }}
          >
            Paste an article or a short claim, submit it, and see the model’s prediction and confidence, visualised with a clean ring
            breakdown.
          </Typography>
        </Box>

        <Divider sx={{ borderColor: 'rgba(0,0,0,0.10)', mb: 3 }} />

        {/* Grid of sections */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
            gap: 2,
          }}
        >
          {/* Inner cards: slightly lighter than outer */}
          <Paper
            variant="outlined"
            sx={{
              p: 2.5,
              borderRadius: 3,
              bgcolor: '#f6f6f6',
              borderColor: 'rgba(0,0,0,0.12)',
            }}
          >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
              Purpose
            </Typography>
            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.65 }}>
              This app is a lightweight interface for experimenting with misinformation detection. It’s designed to make
              model outputs easy to interpret. It returns a label, as well as how confident the model is, showing which alternatives it
              considered.
            </Typography>
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              p: 2.5,
              borderRadius: 3,
              bgcolor: '#f6f6f6',
              borderColor: 'rgba(0,0,0,0.12)',
            }}
          >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
              How to use
            </Typography>
            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.65 }}>
              Use the navigation bar to switch between <b>Check Article</b> and <b>Check Claim</b>. Paste text, hit
              submit, and the prediction panel will render the result. For claims, you’ll also see the top competing
              labels and an <b>Other</b> bucket.
            </Typography>
          </Paper>

          {/* Articles segment */}
            <Paper
            variant="outlined"
            sx={{
                p: 2.5,
                borderRadius: 3,
                bgcolor: '#f6f6f6',
                borderColor: 'rgba(0,0,0,0.12)',
            }}
            >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
                Articles
            </Typography>

            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.65 }}>
                <b>Dataset:</b>{' '}
                <Link
                href="https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset"
                target="_blank"
                rel="noreferrer"
                underline="hover"
                sx={{
                    color: '#0b57d0',
                    fontWeight: 700,
                    textDecorationColor: 'rgba(11,87,208,0.45)',
                }}
                >
                ISOT Fake News Dataset
                </Link>
                <br />
                <b>Model:</b> <span style={{ opacity: 0.75 }}>Describe the architecture + training approach here</span>
                <br />
                <b>Labels:</b> <span style={{ opacity: 0.75 }}>e.g. Real/Fake or True/False</span>
            </Typography>

            <Typography sx={{ mt: 1, color: 'rgba(0,0,0,0.55)', fontSize: 13, lineHeight: 1.55 }}>
                Tip: Include preprocessing steps (tokenization, max length), training metrics, and any calibration.
            </Typography>
            </Paper>

            {/* Claims segment */}
            <Paper
            variant="outlined"
            sx={{
                p: 2.5,
                borderRadius: 3,
                bgcolor: '#f6f6f6',
                borderColor: 'rgba(0,0,0,0.12)',
            }}
            >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
                Claims
            </Typography>

            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.65 }}>
                <b>Dataset:</b>{' '}
                <Link
                href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset"
                target="_blank"
                rel="noreferrer"
                underline="hover"
                sx={{
                    color: '#0b57d0',
                    fontWeight: 700,
                    textDecorationColor: 'rgba(11,87,208,0.45)',
                }}
                >
                LIAR Dataset
                </Link>
                <br />
                <b>Model:</b> <span style={{ opacity: 0.75 }}>Describe the architecture + training approach here</span>
                <br />
                <b>Labels:</b> <span style={{ opacity: 0.75 }}>e.g. LIAR-style multi-class labels</span>
            </Typography>

            <Typography sx={{ mt: 1, color: 'rgba(0,0,0,0.55)', fontSize: 13, lineHeight: 1.55 }}>
                Tip: Mention your top classes, class imbalance handling, and evaluation (macro-F1 is useful here).
            </Typography>
            </Paper>


          <Paper
            variant="outlined"
            sx={{
              p: 2.5,
              borderRadius: 3,
              bgcolor: '#f6f6f6',
              borderColor: 'rgba(0,0,0,0.12)',
            }}
          >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
              Models used
            </Typography>
            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.65 }}>
              <b>Articles:</b> Binary Classification with a probability breakdown.
              <br />
              <b>Claims:</b> Multi-class Classification (top-3 labels shown + <b>Other</b>).
              <br />
              Outputs are visualised directly from the probability vector returned by the API.
            </Typography>
            <Typography sx={{ mt: 1, color: 'rgba(0,0,0,0.55)', fontSize: 13, lineHeight: 1.55 }}>
              You can expand this section later with the exact architecture, datasets, and training setup.
            </Typography>
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              p: 2.5,
              borderRadius: 3,
              bgcolor: '#f6f6f6',
              borderColor: 'rgba(0,0,0,0.12)',
            }}
          >
            <Typography sx={{ fontWeight: 900, fontSize: 18, mb: 0.75 }}>
              Planned improvements
            </Typography>
            <Typography sx={{ color: 'rgba(0,0,0,0.75)', lineHeight: 1.75 }}>
              • Better explainability (highlight key phrases that influenced the prediction)
              <br />
              • Calibration + confidence warnings for uncertain outputs
              <br />
              • Dataset expansion + robustness checks across domains
              <br />
              • Take user feedback into account for continual learning
              <br />
              • Mobile optimisation
            </Typography>
          </Paper>
        </Box>

        <Divider sx={{ borderColor: 'rgba(0,0,0,0.10)', my: 3 }} />

        <Box sx={{ textAlign: 'center' }}>
          <Typography sx={{ color: 'rgba(0,0,0,0.65)', fontSize: 14 }}>
            Tip: Try a short claim vs a longer article to see the different prediction modes.
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}

export default Home
