import type { FormEvent, Dispatch, SetStateAction } from 'react'
import { Typography, TextField, Box, Button, Paper } from '@mui/material'

interface CheckArticleProps {
  submitArticle: (e: FormEvent) => Promise<void>
  article: string
  setArticle: Dispatch<SetStateAction<string>>
}

const CheckArticle = ({ submitArticle, article, setArticle }: CheckArticleProps) => {
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 3,
        height: '100%',
        bgcolor: '#e0e0e0',
        color: '#111',
        borderColor: 'rgba(0,0,0,0.18)',
        borderRadius: 3,
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
        Check Article
      </Typography>

      <Box
        component="form"
        onSubmit={submitArticle}
        sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
      >
        <TextField
          label="Article text"
          placeholder="Paste an article here…"
          value={article}
          onChange={(e) => setArticle(e.target.value)}
          multiline
          rows={18}
          fullWidth
          sx={{
            '& .MuiOutlinedInput-root': {
              bgcolor: '#f6f6f6',
              borderRadius: 2,
            },
            '& textarea': {
              overflowY: 'auto',
            },
          }}
        />

        <Button
          type="submit"
          variant="contained"
          size="large"
          sx={{ textTransform: 'none', fontWeight: 700, alignSelf: 'flex-start' }}
        >
          Submit article
        </Button>
      </Box>
    </Paper>
  )
}

export default CheckArticle
