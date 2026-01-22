import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from "recharts";

type Props = {
  series: Array<{ date: string; close: number; prob_up: number }>;
};

export default function Charts({ series }: Props) {
  return (
    <div style={{ width: "100%", height: 360 }}>
      <ResponsiveContainer>
        <LineChart data={series}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" hide />
          <YAxis yAxisId="left" domain={["auto", "auto"]} />
          <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="close" dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="prob_up" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
