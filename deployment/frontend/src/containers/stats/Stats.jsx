import { Box, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import commonStyles from "@/styles/commonStyles";

const Stats = ({ model }) => {
    const [text, setText] = useState("");
    useEffect(() => {
        const updateText = async () => {
            const response = await fetch(
                `/evaluations/${model}/evaluation.txt`
            );
            if (response.status == 200) {
                const text = await response.text();
                setText(text);
            }
        };
        updateText();
    }, [model]);
    return (
        <Box sx={commonStyles.sectionContainer}>
            <Typography variant="h3">Test Set Performance</Typography>
            <Typography variant="h4">Report</Typography>
            <Typography variant="body1" sx={commonStyles.codeBlock}>
                {text}
            </Typography>
            <Typography variant="h4">Visualizations</Typography>
            <img src={`/evaluations/${model}/evaluation.jpg`} />
        </Box>
    );
};

export default Stats;
