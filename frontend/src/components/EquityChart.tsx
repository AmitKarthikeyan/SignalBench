import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from "recharts";

type Props = {
  series: Array<{ date: string; equity: number; buy_hold: number }>;
};

export default function EquityChart({ series }: Props) {
  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={series}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" hide />
          <YAxis domain={["auto","auto"]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="equity" dot={false} />
          <Line type="monotone" dataKey="buy_hold" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
