const predictArticle = async (article: string) => {
    const response = await fetch(`/api/predict/article`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ article })
    })
    return response.json()
}

const predictClaim = async (claim: string) => {
    const response = await fetch(`/api/predict/claim`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ claim })
    })
    return response.json()
   
}

export default { predictArticle, predictClaim}