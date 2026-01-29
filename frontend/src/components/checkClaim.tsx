import type { FormEvent, Dispatch, SetStateAction } from 'react'
import { Typography, TextField, Box, Button, Paper } from '@mui/material'

interface CheckClaimProps {
  submitClaim: (e: FormEvent) => Promise<void>
  claim: string
  setClaim: Dispatch<SetStateAction<string>>
}

const CheckClaim = ({ submitClaim, claim, setClaim }: CheckClaimProps) => {
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
        Check Claim
      </Typography>

      <Box
        component="form"
        onSubmit={submitClaim}
        sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
      >
        <TextField
          label="Claim text"
          placeholder="Paste a claim here…"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          multiline
          rows={14}
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
          Submit claim
        </Button>
      </Box>
    </Paper>
  )
}

export default CheckClaim
